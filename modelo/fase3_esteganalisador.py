import os
import csv
import time
import random
import shutil
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

import fase0_codificador_decodificador as base
import fase2_wgan_gp as fase2
import fase25_residual as fase25


CONFIG = dict(fase2.CONFIG)
CONFIG.update({

    "arquitetura": "residual",

    "alvo_psnr": None,

    "lambda_srnet": 0.01,
    "n_srnet": 1,
    "alvo_srnet": "equilibrio",
    "lr_srnet": 1e-4,
    "srnet_compacta": False,

    "epocas": 60,
    "usar_amp": False,
    "carregar_fase0": None,
    "carregar_fase2": "runs/phase2_residual/ckpt_best.pt",


    "retomar_de": None,

    "salvar_otimizadores": True,
    "pasta_saida": "runs/phase3_bossbase",

    "espelho_ckpt": None,     
})

COLAB_GPU = dict(fase2.COLAB_GPU)
COLAB_GPU.update({
    "epocas": 60,
    "lambda_srnet": 0.01,
    "usar_amp": False,

    "num_trabalhadores": max(1, min(4, (os.cpu_count() or 2) - 1)),
})


class _Tipo1(nn.Module):
    def __init__(self, entrada, saida):
        super().__init__()
        self.conv = nn.Conv2d(entrada, saida, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(saida)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class _Tipo2(nn.Module):
    def __init__(self, canais):
        super().__init__()
        self.c1 = nn.Conv2d(canais, canais, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(canais)
        self.c2 = nn.Conv2d(canais, canais, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(canais)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)), inplace=True)
        h = self.b2(self.c2(h))
        return x + h


class _Tipo3(nn.Module):
    def __init__(self, entrada, saida):
        super().__init__()
        self.c1 = nn.Conv2d(entrada, saida, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(saida)
        self.c2 = nn.Conv2d(saida, saida, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(saida)
        self.pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.atalho_conv = nn.Conv2d(entrada, saida, 1, stride=2, bias=False)
        self.atalho_bn = nn.BatchNorm2d(saida)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)), inplace=True)
        h = self.pool(self.b2(self.c2(h)))
        return self.atalho_bn(self.atalho_conv(x)) + h


class _Tipo4(nn.Module):
    def __init__(self, entrada, saida):
        super().__init__()
        self.c = nn.Conv2d(entrada, saida, 3, padding=1, bias=False)
        self.b = nn.BatchNorm2d(saida)

    def forward(self, x):
        h = F.relu(self.b(self.c(x)), inplace=True)
        return h.mean(dim=(2, 3))


class SRNet(nn.Module):
    """Imagem -> 1 logit. Alto = "tem mensagem escondida" (estego)."""
    def __init__(self, canais, compacta=False):
        super().__init__()
        e = 0.5 if compacta else 1.0
        f = lambda n: max(4, int(n * e))

        self.tipo1 = nn.Sequential(_Tipo1(canais, f(64)), _Tipo1(f(64), f(16)))
        self.tipo2 = nn.Sequential(*[_Tipo2(f(16)) for _ in range(5)])
        self.tipo3 = nn.Sequential(
            _Tipo3(f(16), f(16)),
            _Tipo3(f(16), f(64)),
            _Tipo3(f(64), f(128)),
            _Tipo3(f(128), f(256)),
        )
        self.tipo4 = _Tipo4(f(256), f(512))
        self.para_logit = nn.Linear(f(512), 1)

    def forward(self, x):
        h = self.tipo1(x)
        h = self.tipo2(h)
        h = self.tipo3(h)
        h = self.tipo4(h)
        return self.para_logit(h).squeeze(1)

auc_binaria = fase2.auc_binaria


@torch.no_grad()
def avaliar_srnet(codificador, srnet, loader, cfg):

    dispositivo = cfg["dispositivo"]
    codificador.eval(); srnet.eval()
    todos_scores, todos_rotulos = [], []

    for portadora in loader:
        portadora = portadora.to(dispositivo, non_blocking=True)
        b = portadora.size(0)
        mensagem = base.mensagem_aleatoria(b, cfg)
        estego = codificador(portadora, base.espalhar_mensagem(mensagem, cfg))

        logits = srnet(torch.cat([portadora, estego], dim=0)).cpu()
        todos_scores.append(logits)
        todos_rotulos.append(torch.cat([torch.zeros(b), torch.ones(b)]))

    scores = torch.cat(todos_scores)
    rotulos = torch.cat(todos_rotulos)
    auc = auc_binaria(scores, rotulos)

    predito = (scores > 0).float()
    vp = float(((predito == 1) & (rotulos == 1)).sum())
    fp = float(((predito == 1) & (rotulos == 0)).sum())
    fn = float(((predito == 0) & (rotulos == 1)).sum())
    acc = (predito == rotulos).float().mean().item()
    precisao = vp / (vp + fp) if (vp + fp) > 0 else float("nan")
    recall = vp / (vp + fn) if (vp + fn) > 0 else float("nan")
    f1 = (2 * precisao * recall / (precisao + recall)
          if (precisao == precisao and recall == recall and precisao + recall > 0)
          else float("nan"))
    return {"auc": auc, "acuracia_deteccao": acc, "precisao": precisao,
            "recall": recall, "f1": f1}


def _carregar_gerador(caminho, codificador, decodificador, critico, dispositivo, log=print):

    if not caminho or not os.path.exists(caminho):
        log(f"[warm start] {caminho!r} não encontrado -> gerador DO ZERO.\n"
              "             A Fase 3 é para ser construída SOBRE a Fase 2. Rode a Fase 2 antes,\n"
              "             ou aponte 'carregar_fase2' para o ckpt_best.pt no Drive.")
        return None

    estado = torch.load(caminho, map_location=dispositivo)
    cod = estado.get("codificador", estado.get("encoder"))
    dec = estado.get("decodificador", estado.get("decoder"))
    if cod is None or dec is None:
        raise KeyError(f"{caminho}: não achei os pesos do codificador/decodificador. "
                       f"Chaves presentes: {list(estado.keys())}")
    try:
        codificador.load_state_dict(cod)
        decodificador.load_state_dict(dec)
    except RuntimeError as e:
        raise RuntimeError(f"{caminho}: os pesos não cabem no gerador '{fase25._ATUAL}'. "
                           "Confira cfg['arquitetura'] — checkpoint denso só carrega em gerador "
                           "denso, residual só em residual.") from e

    if "critico" in estado:
        critico.load_state_dict(estado["critico"])
        log(f"[warm start] gerador + crítico carregados de {caminho} (época {estado.get('epoca', '?')})")
    else:
        log(f"[warm start] gerador carregado de {caminho}; crítico DO ZERO "
              "(checkpoint não tinha os pesos do crítico)")
    return estado.get("val", {}).get("psnr")


class _Log:
 
    def __init__(self, caminho):
        self.caminho = caminho
        self.arquivo = open(caminho, "a", encoding="utf-8")

    def __call__(self, texto=""):
        print(texto)
        self.arquivo.write(texto + "\n")
        self.arquivo.flush()

    def fechar(self):
        self.arquivo.close()


def _espelhar(cfg, origem, log=print):
    destino = cfg.get("espelho_ckpt")
    if not destino:
        return
  
    if os.path.abspath(destino) == os.path.abspath(cfg["pasta_saida"]):
        return
    try:
        os.makedirs(destino, exist_ok=True)
        shutil.copy2(origem, os.path.join(destino, os.path.basename(origem)))
    except Exception as e:
        log(f"[espelho] falhou ao copiar para {destino}: {e}")


def treinar():
    cfg = CONFIG
    torch.manual_seed(cfg["semente"]); random.seed(cfg["semente"])
    os.makedirs(cfg["pasta_saida"], exist_ok=True)
    dispositivo = cfg["dispositivo"]
    tm = base._tamanho_msg(cfg)
    bpp = cfg["profundidade_dados"] * (tm / cfg["tamanho_img"]) ** 2

    log = _Log(os.path.join(cfg["pasta_saida"], "treino.log"))
    log(f"===== Fase 3 iniciada em {datetime.now():%Y-%m-%d %H:%M:%S} =====")
    log("[cfg] " + "  ".join(f"{k}={cfg[k]}" for k in
        ("tamanho_img", "tamanho_msg", "tamanho_lote", "largura", "epocas",
         "lambda_imagem", "lambda_mensagem", "lambda_residuo", "lambda_adv",
         "lambda_srnet", "n_critic", "n_srnet", "lr_srnet", "srnet_compacta", "arquitetura")))
    log(f"[infra] dispositivo={dispositivo}  payload={bpp:.4g} bpp  "
          f"lambda_adv={cfg['lambda_adv']}  lambda_srnet={cfg['lambda_srnet']}  "
          f"n_critic={cfg['n_critic']}  n_srnet={cfg['n_srnet']}"
          + ("  [SRNet COMPACTA]" if cfg["srnet_compacta"] else ""))

    loader_treino, loader_val = base.montar_loaders(cfg)

    fase25.usar(cfg["arquitetura"])
    codificador = base.Codificador(cfg["canais"], cfg["profundidade_dados"], cfg["largura"]).to(dispositivo)
    decodificador = base.Decodificador(cfg["canais"], cfg["profundidade_dados"], cfg["largura"]).to(dispositivo)
    critico = fase2.Critico(cfg["canais"], cfg["largura_critico"]).to(dispositivo)
    srnet = SRNet(cfg["canais"], compacta=cfg["srnet_compacta"]).to(dispositivo)

    psnr_partida = _carregar_gerador(cfg.get("carregar_fase2"), codificador, decodificador,
                                     critico, dispositivo, log=log)
    if cfg["alvo_psnr"] is not None:
        alvo_psnr = cfg["alvo_psnr"]
        log(f"[alvo  ] critério (2): PSNR >= {alvo_psnr:.2f} dB (fixado em cfg)")
    elif psnr_partida is not None:
        alvo_psnr = round(psnr_partida - 1.0, 2)
        log(f"[alvo  ] critério (2): PSNR >= {alvo_psnr:.2f} dB (partida {psnr_partida:.2f} - 1 dB)")
    else:
        alvo_psnr = 37.27
        log(f"[alvo  ] checkpoint sem PSNR gravado -> critério (2) no padrão da Fase 2.5: "
            f"PSNR >= {alvo_psnr:.2f} dB")
    n_par = sum(p.numel() for p in srnet.parameters())
    log(f"[srnet] iniciada DO ZERO — {n_par/1e6:.2f}M parâmetros")

    opt_g = torch.optim.Adam(list(codificador.parameters()) + list(decodificador.parameters()),
                             lr=cfg["taxa_aprend"], betas=(cfg["beta1"], cfg["beta2"]))
    opt_c = torch.optim.Adam(critico.parameters(),
                             lr=cfg["lr_critico"], betas=(cfg["beta1"], cfg["beta2"]))
    opt_s = torch.optim.Adam(srnet.parameters(), lr=cfg["lr_srnet"])

    melhor = {"score": float("-inf"), "epoca": 0, "acuracia_bits": 0.0, "psnr": 0.0,
              "ssim": 0.0, "auc": float("nan")}
    epoca_inicial = 1
    ck_ret = cfg.get("retomar_de")
    if ck_ret and os.path.exists(ck_ret):
        est = torch.load(ck_ret, map_location=dispositivo)
        codificador.load_state_dict(est["codificador"])
        decodificador.load_state_dict(est["decodificador"])
        critico.load_state_dict(est["critico"])
        srnet.load_state_dict(est["srnet"])
        if "opt_g" in est:
            opt_g.load_state_dict(est["opt_g"])
            opt_c.load_state_dict(est["opt_c"])
            opt_s.load_state_dict(est["opt_s"])
        else:
            log("[retomada] o checkpoint não tem o estado dos otimizadores: o Adam recomeça "
                "do zero e as primeiras épocas dão um solavanco.")
        if est.get("melhor"):
            melhor = est["melhor"]
        if est.get("rng") is not None:
            torch.set_rng_state(est["rng"].cpu())
        epoca_inicial = int(est["epoca"]) + 1
        log(f"[retomada] {ck_ret} — continuando da época {epoca_inicial} "
            f"(melhor até aqui: época {melhor['epoca']}, PSNR {melhor['psnr']:.2f} dB)")
        if epoca_inicial > cfg["epocas"]:
            log(f"[retomada] o run já tinha {est['epoca']} épocas e cfg['epocas'] = "
                f"{cfg['epocas']}. Suba 'epocas' para continuar.")
    elif ck_ret:
        log(f"[retomada] {ck_ret!r} não encontrado -> começando do zero.")

    retomando = epoca_inicial > 1
    caminho_csv = os.path.join(cfg["pasta_saida"], "metricas_val.csv")
    if not (retomando and os.path.exists(caminho_csv)):
        with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoca", "acuracia_bits", "psnr", "ssim", "auc_srnet",
                                    "auc_critico", "w_dist", "acuracia_det", "precisao",
                                    "recall", "f1"])

   
    caminho_csv_treino = os.path.join(cfg["pasta_saida"], "metricas_treino.csv")
    if not (retomando and os.path.exists(caminho_csv_treino)):
        with open(caminho_csv_treino, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoca", "perda_c", "w_dist", "adv", "srnet_bce", "enganar",
                                    "img_mse", "msg_bce", "acuracia_bits"])

    t_inicio = time.time()
    for epoca in range(epoca_inicial, cfg["epocas"] + 1):
        t_epoca = time.time()
        codificador.train(); decodificador.train(); critico.train(); srnet.train()
        ac = {"c": 0.0, "w": 0.0, "adv": 0.0, "s": 0.0, "eng": 0.0,
              "img": 0.0, "msg": 0.0, "acc": 0.0, "n": 0}

        for portadora in loader_treino:
            portadora = portadora.to(dispositivo, non_blocking=True)
            b = portadora.size(0)

            for _ in range(cfg["n_critic"]):
                mensagem = base.mensagem_aleatoria(b, cfg)
                with torch.no_grad():
                    estego = codificador(portadora, base.espalhar_mensagem(mensagem, cfg))
                score_real = critico(portadora).mean()
                score_falsa = critico(estego).mean()
                gp = fase2.penalidade_gradiente(critico, portadora, estego, dispositivo)
                perda_c = score_falsa - score_real + cfg["lambda_gp"] * gp
                opt_c.zero_grad(); perda_c.backward(); opt_c.step()

            w_dist = (score_real - score_falsa).item()

         
            for _ in range(cfg["n_srnet"]):
                mensagem = base.mensagem_aleatoria(b, cfg)
                with torch.no_grad():
                    estego = codificador(portadora, base.espalhar_mensagem(mensagem, cfg))
                logits_s = srnet(torch.cat([portadora, estego], dim=0))
                alvos_s = torch.cat([torch.zeros(b, device=dispositivo),
                                     torch.ones(b, device=dispositivo)])
                perda_s = F.binary_cross_entropy_with_logits(logits_s, alvos_s)
                opt_s.zero_grad(); perda_s.backward(); opt_s.step()

            mensagem = base.mensagem_aleatoria(b, cfg)
            estego = codificador(portadora, base.espalhar_mensagem(mensagem, cfg))
            logits = base.agregar_logits(decodificador(estego), cfg)

            perda_img = F.mse_loss(estego, portadora)
            perda_msg = F.binary_cross_entropy_with_logits(logits, mensagem)
            perda_res = (estego - portadora).abs().mean()
            perda_adv = -critico(estego).mean()

  
            logits_g = srnet(torch.cat([portadora.detach(), estego], dim=0))
            logit_est_g = logits_g[b:]
            valor_alvo = 0.5 if cfg["alvo_srnet"] == "equilibrio" else 0.0
            perda_eng = F.binary_cross_entropy_with_logits(
                logit_est_g, torch.full_like(logit_est_g, valor_alvo))

            perda_g = (cfg["lambda_imagem"] * perda_img
                       + cfg["lambda_mensagem"] * perda_msg
                       + cfg["lambda_residuo"] * perda_res
                       + cfg["lambda_adv"] * perda_adv
                       + cfg["lambda_srnet"] * perda_eng)
            opt_g.zero_grad(); perda_g.backward(); opt_g.step()

            ac["c"] += perda_c.item() * b;    ac["w"] += w_dist * b
            ac["adv"] += perda_adv.item() * b; ac["s"] += perda_s.item() * b
            ac["eng"] += perda_eng.item() * b
            ac["img"] += perda_img.item() * b; ac["msg"] += perda_msg.item() * b
            ac["acc"] += base.acuracia_bits(logits, mensagem) * b
            ac["n"] += b

        n = ac["n"]
        seg = time.time() - t_epoca
        feitas = epoca - epoca_inicial + 1
        restam = (cfg["epocas"] - epoca) * (time.time() - t_inicio) / feitas
        log(f"[tempo ] época {epoca:3d}  {seg:6.1f}s  ({ac['n']//cfg['tamanho_lote']} passos de gerador)"
            f"  restam ~{restam/60:.0f} min")
        log(f"[treino] época {epoca:3d}  perda_c={ac['c']/n:+.3f}  W={ac['w']/n:+.3f}  "
              f"srnet_bce={ac['s']/n:.4f}  enganar={ac['eng']/n:.4f}  "
              f"img_mse={ac['img']/n:.5f}  msg_bce={ac['msg']/n:.4f}  "
              f"acuracia_bits={ac['acc']/n:.4f}")

        val = base.avaliar(codificador, decodificador, loader_val, cfg)
        det = avaliar_srnet(codificador, srnet, loader_val, cfg)
        auc, auc_ef = det["auc"], max(det["auc"], 1.0 - det["auc"])
        auc_c = fase2.avaliar_critico(codificador, critico, loader_val, cfg)

        c1 = val["acuracia_bits"] >= cfg["alvo_bitacc"]
        c2 = val["psnr"] >= alvo_psnr
        c3 = abs(auc - 0.5) <= 0.05
        marca = "".join(["1" if c1 else ".", "2" if c2 else ".", "3" if c3 else "."])
        log(f"[  val ] época {epoca:3d}  acuracia_bits={val['acuracia_bits']:.4f}  "
              f"psnr={val['psnr']:.2f}dB  ssim={val['ssim']:.4f}  "
              f"auc_srnet={auc:.4f} (efetiva {auc_ef:.4f})  "
              f"auc_critico={auc_c:.4f} (efetiva {max(auc_c, 1 - auc_c):.4f})  "
              f"criterios[{marca}]")

 
        log(f"[detect] época {epoca:3d}  acuracia={det['acuracia_deteccao']:.4f}  "
            f"precisao={det['precisao']:.4f}  recall={det['recall']:.4f}  "
            f"f1={det['f1']:.4f}")

        with open(caminho_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoca, f"{val['acuracia_bits']:.4f}", f"{val['psnr']:.2f}",
                                    f"{val['ssim']:.4f}", f"{auc:.4f}", f"{auc_c:.4f}",
                                    f"{ac['w']/n:.4f}", f"{det['acuracia_deteccao']:.4f}",
                                    f"{det['precisao']:.4f}", f"{det['recall']:.4f}",
                                    f"{det['f1']:.4f}"])
        with open(caminho_csv_treino, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoca, f"{ac['c']/n:.4f}", f"{ac['w']/n:.4f}",
                                    f"{ac['adv']/n:.4f}", f"{ac['s']/n:.4f}", f"{ac['eng']/n:.4f}",
                                    f"{ac['img']/n:.6f}", f"{ac['msg']/n:.4f}", f"{ac['acc']/n:.4f}"])


        for reg in (caminho_csv, caminho_csv_treino, os.path.join(cfg["pasta_saida"], "treino.log")):
            _espelhar(cfg, reg, log)

      
        score = (val["psnr"] if c1 else val["acuracia_bits"] - 100.0)
        if score > melhor["score"]:
            melhor = {"score": score, "epoca": epoca, "auc": auc, "det": det,
                      **{k: val[k] for k in ("acuracia_bits", "psnr", "ssim")}}
            destino = os.path.join(cfg["pasta_saida"], "ckpt_best.pt")
            torch.save({"codificador": codificador.state_dict(),
                        "decodificador": decodificador.state_dict(),
                        "critico": critico.state_dict(),
                        "srnet": srnet.state_dict(),
                        "epoca": epoca, "cfg": cfg, "val": val, "det": det},
                       destino)
            _espelhar(cfg, destino, log)

        if epoca % cfg["salvar_cada"] == 0:
            base.salvar_amostras(codificador, loader_val, cfg, epoca)
            destino = os.path.join(cfg["pasta_saida"], "ckpt_last.pt")
            estado = {"codificador": codificador.state_dict(),
                      "decodificador": decodificador.state_dict(),
                      "critico": critico.state_dict(),
                      "srnet": srnet.state_dict(),
                      "epoca": epoca, "cfg": cfg,
                      "melhor": melhor, "rng": torch.get_rng_state()}
            if cfg.get("salvar_otimizadores", True):
                estado.update(opt_g=opt_g.state_dict(), opt_c=opt_c.state_dict(),
                              opt_s=opt_s.state_dict())
            torch.save(estado, destino)
            _espelhar(cfg, destino, log)

    auc_ef = max(melhor["auc"], 1.0 - melhor["auc"])
    log(f"\n[fim] checkpoints, amostras e metricas_val.csv em: {cfg['pasta_saida']}")
    log(f"[melhor] época {melhor['epoca']}: acuracia_bits={melhor['acuracia_bits']:.4f}  "
          f"psnr={melhor['psnr']:.2f}dB  ssim={melhor['ssim']:.4f}  auc_srnet={melhor['auc']:.4f}")
    log("\nCRITÉRIOS DE SAÍDA DA FASE 3:")
    log(f"  (1) acuracia_bits >= 0.99          -> {melhor['acuracia_bits']:.4f}  "
          f"{'OK' if melhor['acuracia_bits'] >= 0.99 else 'FALHOU: baixe lambda_srnet'}")
    log(f"  (2) PSNR >= {alvo_psnr:.2f} dB (partida - 1dB) -> {melhor['psnr']:.2f}  "
          f"{'OK' if melhor['psnr'] >= alvo_psnr else 'FALHOU: baixe lambda_srnet ou suba lambda_imagem'}")
    if melhor["acuracia_bits"] < cfg["alvo_bitacc"]:
        veredito = ("SEM VALOR: a mensagem não é recuperada — o gerador engana o detector "
                    "deixando de esconder. Baixe lambda_srnet")
    elif abs(melhor["auc"] - 0.5) <= 0.05:
        veredito = "OK"
    elif melhor["auc"] > 0.5:
        veredito = "detector separando: suba lambda_srnet"
    else:
        veredito = ("gerador PASSOU DO PONTO (estego com score menor que portadora). "
                    "NÃO suba lambda_srnet — isso piora. Use alvo_srnet='equilibrio'")
    log(f"  (3) AUC do laço ~ 0.5              -> {melhor['auc']:.4f} (efetiva {auc_ef:.4f})  {veredito}")
    d = melhor.get("det") or {}
    if d:
        log(f"      detector na melhor época: acuracia={d['acuracia_deteccao']:.4f}  "
            f"precisao={d['precisao']:.4f}  recall={d['recall']:.4f}  f1={d['f1']:.4f}")
        if d["recall"] > 0.95 and d["precisao"] < 0.55:
            log("      ATENÇÃO: recall ~1 com precisão ~0,5 = o detector responde 'estego' "
                "para tudo. Isso é detector que NAO CONVERGIU, nao seguranca.")
    log("\nATENÇÃO: o critério (3) mede o detector que participou do treino. Ele NÃO prova")
    log("segurança — pode significar que aquela SRNet não convergiu. A prova é a Fase 4,")
    log("que treina um esteganalisador independente e mede P_E. Ver ROADMAP.md.")
    log(f"\n[registros] treino.log, metricas_val.csv e metricas_treino.csv em {cfg['pasta_saida']}")
    log(f"[retomada] se a sessão cair, rode de novo com "
        f"cfg['retomar_de'] = '{os.path.join(cfg['pasta_saida'], 'ckpt_last.pt')}'")
    if not cfg.get("espelho_ckpt"):
        log("\nLEMBRETE: 'espelho_ckpt' está vazio. Se isto rodou no Colab, BAIXE AGORA o")
        log("ckpt_best.pt E os arquivos de registro — foi assim que os pesos das Fases 0 e 2")
        log("se perderam, e a saída da célula do notebook some junto com o runtime.")
    else:
        _espelhar(cfg, os.path.join(cfg["pasta_saida"], "treino.log"), log)
        log(f"[espelho] registros e checkpoints copiados para {cfg['espelho_ckpt']}")
    log.fechar()
    return {"acuracia_bits": melhor["acuracia_bits"], "psnr": melhor["psnr"],
            "ssim": melhor["ssim"], "auc": melhor["auc"], "alvo_psnr": alvo_psnr}


if __name__ == "__main__":
    treinar()
