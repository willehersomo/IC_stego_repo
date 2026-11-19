"""Fase 4.0 — régua externa: P_E com esteganalisadores treinados do zero (09/11/2026)."""
import csv
import math
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fase0_codificador_decodificador as base
import fase3_esteganalisador as f3
import fase30_custo_wgan as f30


CONFIG = {
    "pasta_dados": None,
    "conjunto": "boss",
    "tamanho_img": 128,
    "canais": 1,
    "semente_split": 42,
    "fracao_val": 0.05,
    "n_val_detector": 1000,
    "max_imagens": None,
    "semente_embutir": 7,
    "epocas": 20,
    "pares_por_lote": 32,
    "lr": {"srnet": 1e-3, "srnet_cur": 1e-3, "effnet": 5e-4, "effnet_cur": 5e-4},
    "lr_fino": 5e-4,
    "curriculo": {"srnet_cur": (1.0, 0.4, 0.2),
                  "effnet_cur": (1.0, 0.4, 0.2)},
    "epocas_curriculo": 5,
    "decaimento": 1e-4,
    "usar_amp": True,
    "razao_taxa_max": 1.05,
    "salvar_pesos": True,
    "limite_horas": None,
    "pasta_saida": "runs/phase40",
    "espelho": None,
    "dispositivo": "cuda" if torch.cuda.is_available() else "cpu",
}



pe_auc = f30.pe_auc


def quarteto(scores, rotulos):
    s, y = np.asarray(scores), np.asarray(rotulos)
    p = (s > 0).astype(np.int64)
    vp, fp, fn = ((p == 1) & (y == 1)).sum(), ((p == 1) & (y == 0)).sum(), ((p == 0) & (y == 1)).sum()
    prec = vp / (vp + fp) if vp + fp else float("nan")
    rec = vp / (vp + fn) if vp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else float("nan")
    return {"acuracia": float((p == y).mean()), "precisao": float(prec),
            "recall": float(rec), "f1": float(f1)}


@torch.no_grad()
def metricas_imagem(capas, estego, lote=100):
    """PSNR (da média dos MSE), SSIM (o da Fase 0, média por imagem), MSE, taxa e RMS, em uint8."""
    x, y = capas.float(), estego.float()
    d = y - x
    ss = sum(base.ssim(y[i:i + lote] / 255.0, x[i:i + lote] / 255.0, x.shape[1]) * len(x[i:i + lote])
             for i in range(0, len(x), lote)) / len(x)
    mse = d.pow(2).mean().item()
    return {"psnr": float(10 * math.log10(255.0 ** 2 / max(mse, 1e-12))),
            "ssim": float(ss), "mse": float(mse),
            "taxa": float((d != 0).float().mean()),
            "rms": float(math.sqrt(mse))}



def dividir(cfg):
    """O split da Fase 3.0: (caminhos, treino_det, val_det, teste)."""
    c = dict(f30.CONFIG_SPLIT, pasta_dados=cfg["pasta_dados"], semente_split=cfg["semente_split"],
             fracao_val=cfg["fracao_val"], n_val_detector=cfg["n_val_detector"],
             max_imagens=cfg["max_imagens"])
    return f30.dividir(c)


@torch.no_grad()
def carregar_capas(caminhos, cfg, log=print):
    """Portadoras uint8 (N, C, H, W) com a transformação de VALIDAÇÃO (a do teste do gerador)."""
    cache = os.path.join(cfg["pasta_saida"], "corpus",
                         f"capas_{cfg['canais']}x{cfg['tamanho_img']}_{len(caminhos)}.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    ds = base.DatasetImagens(caminhos, cfg["canais"], cfg["tamanho_img"], treino=False)
    t0 = time.time()
    capas = torch.stack([torch.round(ds[i] * 255.0).to(torch.uint8) for i in range(len(ds))])
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save(capas, cache)
    log(f"[corpus] {len(capas)} portadoras em {time.time() - t0:.0f}s")
    return capas


def _lsbm(capas, bpp, semente=7):
    """LSB matching sem custo (só para o currículo da SRNet): `bpp` bits em posições sorteadas."""
    g = torch.Generator().manual_seed(semente)
    n, amostras = capas.shape[0], capas[0].numel()
    usados = int(round(min(bpp, 1.0) * amostras))
    x = capas.reshape(n, -1).to(torch.int16).clone()
    for i in range(n):
        pos = torch.randperm(amostras, generator=g)[:usados]
        muda = torch.rand(usados, generator=g) < 0.5
        sinal = torch.where(torch.rand(usados, generator=g) < 0.5, -1, 1).to(torch.int16)
        v = x[i, pos]
        sinal = torch.where(v == 0, 1, torch.where(v == 255, -1, sinal)).to(torch.int16)
        x[i, pos] = v + sinal * muda.to(torch.int16)
    return x.clamp(0, 255).to(torch.uint8).view_as(capas)


@torch.no_grad()
def gerar_estego(metodo, capas, idx_te, bpp, cfg, log=print):
    """metodo = {"nome", "tipo": "hill" | "unif" | "custo", "ckpt"}. Devolve (estego uint8, info)."""
    cache = os.path.join(cfg["pasta_saida"], "corpus",
                         f"estego_{metodo['nome']}_{bpp:g}_{cfg['canais']}x{cfg['tamanho_img']}_{len(capas)}.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    dev, t0 = cfg["dispositivo"], time.time()
    rng = torch.Generator().manual_seed(cfg["semente_embutir"])
    lote = f30._lote_por_tamanho(64, capas)
    ger = f30.carregar_gerador(metodo["ckpt"], dev) if metodo["tipo"] == "custo" else None
    partes = []
    for i in range(0, len(capas), lote):
        x = capas[i:i + lote].to(dev).double()
        if metodo["tipo"] == "hill":
            y = f30.estego_hill(x, bpp, rng)
        elif metodo["tipo"] == "unif":
            y = f30.estego_unif(x, bpp, rng)
        elif metodo["tipo"] == "custo":
            with f30.p_hill_em(torch.float64):
                y = f30.embutir(ger, x.float(), bpp, rng)
        else:
            raise ValueError(f"tipo de método desconhecido: {metodo['tipo']}")
        partes.append(y.float().round().clamp(0, 255).to(torch.uint8).cpu())
    estego = torch.cat(partes)
    info = metricas_imagem(capas[idx_te], estego[idx_te])
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save((estego, info), cache)
    log(f"[corpus] {metodo['nome']} @ {bpp:g} bpp: {len(estego)} estegos em {time.time() - t0:.0f}s | "
        f"teste: PSNR {info['psnr']:.2f}  SSIM {info['ssim']:.5f}  MSE {info['mse']:.4f}  "
        f"taxa {info['taxa']:.2%}")
    return estego, info



class _EffNet(nn.Module):
    """EfficientNet-B0 do timm, ImageNet, 1 logit."""

    def __init__(self, canais=1, pretreinada=True):
        super().__init__()
        import timm
        try:
            self.rede = timm.create_model("efficientnet_b0", pretrained=pretreinada,
                                          in_chans=canais, num_classes=1)
            self.pretreinada = pretreinada
        except Exception as erro:
            print(f"[effnet] pesos do ImageNet indisponíveis ({type(erro).__name__}); do zero")
            self.rede = timm.create_model("efficientnet_b0", pretrained=False,
                                          in_chans=canais, num_classes=1)
            self.pretreinada = False
        self.rede.conv_stem.stride = (1, 1)

    def forward(self, x):
        return self.rede((x - 0.449) / 0.226).squeeze(1)


def construir_detector(nome, canais=1):
    if nome in ("srnet", "srnet_cur"):
        return f3.SRNet(canais, compacta=False)
    if nome in ("effnet", "effnet_cur"):
        return _EffNet(canais)
    raise ValueError(f"detector desconhecido: {nome}")


def _aumentar(x):
    """Rotações de 90° e espelhamento: não mudam a estatística do ruído."""
    k = random.randrange(4)
    if k:
        x = torch.rot90(x, k, dims=(2, 3))
    return torch.flip(x, dims=(3,)) if random.random() < 0.5 else x


def _semear(semente):
    torch.manual_seed(semente); random.seed(semente); np.random.seed(semente)


@torch.no_grad()
def _pontuar(rede, capas, estego, idx, cfg):
    """Notas no formato [capas de idx..., estegos de idx...]; rótulos 0 e 1."""
    dev, amp = cfg["dispositivo"], cfg["usar_amp"] and cfg["dispositivo"] == "cuda"
    rede.eval()
    lote = f30._lote_por_tamanho(128, capas)
    sc, sl = [], []
    for parte in (capas, estego):
        for i in range(0, len(idx), lote):
            b = torch.as_tensor(idx[i:i + lote])
            with torch.autocast("cuda", enabled=amp):
                sc.append(rede(parte[b].to(dev).float() / 255.0).float().cpu())
    s = torch.cat(sc).numpy()
    return s, np.r_[np.zeros(len(idx)), np.ones(len(idx))]


def treinar_detector(nome, capas, estego, idx_tr, idx_va, idx_te, cfg, semente, log=print,
                     inicial=None, epocas=None, rotulo=None):
    """Treina na treino_det, escolhe a época pela val_det e (se idx_te) mede no teste."""
    _semear(semente)
    dev, amp = cfg["dispositivo"], cfg["usar_amp"] and cfg["dispositivo"] == "cuda"
    epocas = epocas or cfg["epocas"]
    rotulo = rotulo or nome
    rede = construir_detector(nome, cfg["canais"]).to(dev)
    if inicial is not None:
        rede.load_state_dict(inicial)
    lr = cfg["lr_fino"] if inicial is not None else cfg["lr"][nome]
    opt = torch.optim.AdamW(rede.parameters(), lr=lr, weight_decay=cfg["decaimento"])
    pl = cfg["pares_por_lote"]
    passos = epocas * (len(idx_tr) // pl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=passos, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    capas_dev, estego_dev = capas.to(dev), estego.to(dev)

    melhor = {"pe": 2.0, "epoca": 0, "estado": None}
    t0 = time.time()
    for epoca in range(1, epocas + 1):
        rede.train()
        perm = np.random.permutation(idx_tr)
        soma, n = 0.0, 0
        for i in range(0, len(perm) - pl + 1, pl):
            b = torch.as_tensor(perm[i:i + pl], device=dev)
            x = _aumentar(torch.cat([capas_dev[b], estego_dev[b]]).float() / 255.0)
            y = torch.cat([torch.zeros(pl, device=dev), torch.ones(pl, device=dev)])
            with torch.autocast("cuda", enabled=amp):
                perda = F.binary_cross_entropy_with_logits(rede(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(perda).backward(); scaler.step(opt); scaler.update(); sched.step()
            soma += perda.item(); n += 1
        pe_va, auc_va = pe_auc(*_pontuar(rede, capas, estego, idx_va, cfg))
        marca = ""
        if pe_va <= melhor["pe"]:
            melhor = {"pe": pe_va, "epoca": epoca,
                      "estado": {k: v.detach().cpu().clone() for k, v in rede.state_dict().items()}}
            marca = "  *"
        restam = (epocas - epoca) * (time.time() - t0) / epoca
        log(f"[{rotulo}] época {epoca:2d}  bce={soma / max(n, 1):.4f}  "
            f"val: P_E={pe_va:.4f}  AUC={auc_va:.4f}  restam ~{restam / 60:.0f} min{marca}")
    del capas_dev, estego_dev
    if idx_te is None:
        return {"pe_val": melhor["pe"], "epoca_escolhida": melhor["epoca"], "estado": melhor["estado"]}
    rede.load_state_dict(melhor["estado"])
    notas, rotulos = _pontuar(rede, capas, estego, idx_te, cfg)
    pe, auc = pe_auc(notas, rotulos)
    return {"pe": pe, "auc": auc, "pe_val": melhor["pe"], "epoca_escolhida": melhor["epoca"],
            **quarteto(notas, rotulos), "pretreinada": getattr(rede, "pretreinada", "n/a"),
            "estado": melhor["estado"], "notas": notas.astype(np.float32),
            "minutos": (time.time() - t0) / 60}



def _espelhar_arquivo(local, cfg):
    """Copia um arquivo da pasta de saída para o mesmo caminho relativo no espelho."""
    if not cfg.get("espelho"):
        return
    destino = os.path.join(cfg["espelho"], os.path.relpath(local, cfg["pasta_saida"]))
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    shutil.copy2(local, destino)


def _no_espelho_ou_local(rel, cfg):
    for pasta in (cfg.get("espelho"), cfg["pasta_saida"]):
        if pasta and os.path.exists(os.path.join(pasta, rel)):
            return os.path.join(pasta, rel)
    return None


def preparar_curriculo(nome, capas, idx_tr, idx_va, cfg, semente, log=print):
    """Pré-treino da SRNet em LSBM de carga decrescente; um por semente (sementes independentes)."""
    cargas = cfg["curriculo"][nome]
    rel = os.path.join("curriculo", f"{nome}_{'-'.join(f'{c:g}' for c in cargas)}_"
                       f"{cfg['epocas_curriculo']}ep_s{semente}_{len(capas)}.pt")
    achado = _no_espelho_ou_local(rel, cfg)
    if achado:
        log(f"[curric] {nome} s{semente}: reaproveitado de {achado}")
        return torch.load(achado)
    estado = None
    for carga in cargas:
        estego = _lsbm(capas, carga)
        r = treinar_detector(nome, capas, estego, idx_tr, idx_va, None, cfg, semente, log,
                             inicial=estado, epocas=cfg["epocas_curriculo"],
                             rotulo=f"{nome}@LSBM{carga:g} s{semente}")
        estado = r["estado"]
        aviso = "  <- NÃO APRENDEU (P_E ~0,5)" if r["pe_val"] > 0.45 else ""
        log(f"[curric] {nome} s{semente}: LSBM {carga:g} bpp -> P_E val {r['pe_val']:.4f}{aviso}")
    local = os.path.join(cfg["pasta_saida"], rel)
    os.makedirs(os.path.dirname(local), exist_ok=True)
    torch.save(estado, local)
    _espelhar_arquivo(local, cfg)
    return estado



COLUNAS = ["conjunto", "tamanho", "metodo", "bpp", "detector", "semente", "pe", "auc", "pe_val",
           "acuracia", "precisao", "recall", "f1", "epoca_escolhida", "pretreinada", "curriculo",
           "psnr", "ssim", "mse", "taxa", "razao_taxa_hill", "rms", "epocas", "minutos",
           "n_treino", "n_val", "n_teste"]


def _ler_tabela(caminho):
    if not caminho or not os.path.exists(caminho):
        return []
    with open(caminho, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _chave(conj, tam, metodo, bpp, det, semente):
    return (conj, int(tam), metodo, f"{float(bpp):g}", det, int(semente))


def nome_notas(metodo, bpp, det, semente):
    return os.path.join("notas", f"{metodo}_{bpp:g}_{det}_s{semente}.npy")


def rodar(metodos, cargas=(0.4, 0.2), detectores=("srnet_cur", "effnet"), sementes=(1234, 1235),
          cfg=None):
    """Mede todas as combinações; retomável pela tabela do espelho."""
    cfg = dict(CONFIG, **(cfg or {}))
    os.makedirs(cfg["pasta_saida"], exist_ok=True)
    caminho_log = os.path.join(cfg["pasta_saida"], "fase40.log")
    t_inicio = time.time()

    def log(msg):
        print(msg)
        with open(caminho_log, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log_espelho = _no_espelho_ou_local("fase40.log", cfg)
    if log_espelho and log_espelho != caminho_log and not os.path.exists(caminho_log):
        shutil.copy2(log_espelho, caminho_log)

    tabela_local = os.path.join(cfg["pasta_saida"], "resultados_fase40.csv")
    linhas = _ler_tabela(_no_espelho_ou_local("resultados_fase40.csv", cfg))
    ja = {_chave(l["conjunto"], l["tamanho"], l["metodo"], l["bpp"], l["detector"], l["semente"])
          for l in linhas if int(l.get("epocas", 0)) == cfg["epocas"]}

    caminhos, idx_tr, idx_va, idx_te = dividir(cfg)
    log(f"[split ] {cfg['conjunto']} {cfg['tamanho_img']}²: {len(idx_tr)} treino_det / "
        f"{len(idx_va)} val_det / {len(idx_te)} teste (= val do gerador da Fase 3.0)")
    capas = carregar_capas(caminhos, cfg, log)
    np.save(os.path.join(cfg["pasta_saida"], "idx_teste.npy"), idx_te)
    _espelhar_arquivo(os.path.join(cfg["pasta_saida"], "idx_teste.npy"), cfg)

    nomes = [m["nome"] for m in metodos]
    assert "HILL" in nomes, "o HILL tem de estar entre os métodos: é a referência do ΔP_E"
    total = len(sementes) * len(cargas) * len(detectores) * len(metodos)
    feitos = sum(_chave(cfg["conjunto"], cfg["tamanho_img"], m, b, d, s) in ja
                 for s in sementes for b in cargas for d in detectores for m in nomes)
    log(f"[plano ] {total} treinos ({feitos} já na tabela): sementes {list(sementes)} × cargas "
        f"{list(cargas)} × detectores {list(detectores)} × métodos {nomes}")

    taxa_hill = {}
    for semente in sementes:
        for bpp in cargas:
            for det in detectores:
                pendentes = [m for m in metodos if _chave(cfg["conjunto"], cfg["tamanho_img"], m["nome"],
                                                          bpp, det, semente) not in ja]
                if not pendentes:
                    continue
                if cfg["limite_horas"] and time.time() - t_inicio > cfg["limite_horas"] * 3600:
                    log(f"[parada] limite de {cfg['limite_horas']} h: rode de novo para continuar")
                    return linhas
                inicial = None
                if det in cfg["curriculo"]:
                    inicial = preparar_curriculo(det, capas, idx_tr, idx_va, cfg, semente, log)
                for m in pendentes:
                    if bpp not in taxa_hill:
                        _, ih = gerar_estego(next(x for x in metodos if x["nome"] == "HILL"),
                                             capas, idx_te, bpp, cfg, log)
                        taxa_hill[bpp] = ih["taxa"]
                    estego, info = gerar_estego(m, capas, idx_te, bpp, cfg, log)
                    razao = info["taxa"] / taxa_hill[bpp]
                    if razao > cfg["razao_taxa_max"]:
                        log(f"[ALERTA] {m['nome']} @ {bpp:g} altera {razao:.3f}× o HILL: confira antes de ler o P_E")
                    log("=" * 78)
                    log(f"FASE 4.0 — {m['nome']} @ {bpp:g} bpp × {det} (semente {semente})")
                    log("=" * 78)
                    r = treinar_detector(det, capas, estego, idx_tr, idx_va, idx_te, cfg, semente, log,
                                         inicial=inicial, rotulo=f"{m['nome']}@{bpp:g} {det} s{semente}")
                    estado, notas = r.pop("estado"), r.pop("notas")
                    rel = nome_notas(m["nome"], bpp, det, semente)
                    os.makedirs(os.path.join(cfg["pasta_saida"], "notas"), exist_ok=True)
                    np.save(os.path.join(cfg["pasta_saida"], rel), notas)
                    _espelhar_arquivo(os.path.join(cfg["pasta_saida"], rel), cfg)
                    if cfg["salvar_pesos"]:
                        p = os.path.join(cfg["pasta_saida"], "pesos",
                                         f"{m['nome']}_{bpp:g}_{det}_s{semente}.pt")
                        os.makedirs(os.path.dirname(p), exist_ok=True)
                        torch.save(estado, p)
                        _espelhar_arquivo(p, cfg)
                        os.remove(p)
                    cur = cfg["curriculo"].get(det)
                    linha = {"conjunto": cfg["conjunto"], "tamanho": cfg["tamanho_img"], "metodo": m["nome"],
                             "bpp": f"{bpp:g}", "detector": det, "semente": semente, **r, **info,
                             "razao_taxa_hill": razao,
                             "curriculo": ">".join(f"{c:g}" for c in cur) if cur else "",
                             "epocas": cfg["epocas"], "n_treino": len(idx_tr), "n_val": len(idx_va),
                             "n_teste": len(idx_te)}
                    linhas.append({k: linha.get(k, "") for k in COLUNAS})
                    with open(tabela_local, "w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=COLUNAS); w.writeheader(); w.writerows(linhas)
                    _espelhar_arquivo(tabela_local, cfg)
                    log(f"[teste ] {m['nome']} @ {bpp:g} × {det} s{semente}: P_E={r['pe']:.4f}  "
                        f"AUC={r['auc']:.4f}  (época {r['epoca_escolhida']}, {r['minutos']:.0f} min)")
                    _espelhar_arquivo(caminho_log, cfg)
    log("[fim   ] todas as combinações medidas")
    _espelhar_arquivo(caminho_log, cfg)
    return linhas



def _bootstrap_delta(notas_a, notas_h, n_teste, reamostras, rng, grupos=None):
    """ΔP_E (a − b) em cada reamostra pareada das imagens de teste."""
    y = np.r_[np.zeros(n_teste), np.ones(n_teste)]
    membros = None
    if grupos is not None:
        grupos = np.asarray(grupos)
        assert len(grupos) == n_teste, f"grupos ({len(grupos)}) ≠ imagens de teste ({n_teste})"
        membros = [np.flatnonzero(grupos == g) for g in np.unique(grupos)]
    out = np.empty(reamostras)
    for k in range(reamostras):
        if membros is None:
            i = rng.integers(0, n_teste, n_teste)
        else:
            i = np.concatenate([membros[g] for g in rng.integers(0, len(membros), len(membros))])
        j = np.r_[i, i + n_teste]
        out[k] = pe_auc(notas_a[j], y)[0] - pe_auc(notas_h[j], y)[0]
    return out


def grupos_por_cena(cfg):
    """Cena de cada imagem de teste, na ordem das notas (só nos recortes do DIV2K; no BOSS, None)."""
    cfg = dict(CONFIG, **(cfg or {}))
    sep = f30.CONFIG_SPLIT.get("separador_cena")
    caminhos, _, _, idx_te = dividir(cfg)
    nomes = [os.path.basename(caminhos[i]) for i in idx_te]
    if not sep or not all(sep in n for n in nomes):
        return None
    return np.array([n.split(sep)[0] for n in nomes])


def _ic_pareado(espelho, met_a, met_b, bpp, det, sementes, reamostras, rng, grupos):
    """IC 95% de P_E(met_a) − P_E(met_b), pareado por semente e por imagem (média sobre as sementes)."""
    boots = []
    for s in sementes:
        na = _carregar_notas(espelho, met_a, bpp, det, s)
        nb = _carregar_notas(espelho, met_b, bpp, det, s)
        if na is not None and nb is not None:
            boots.append(_bootstrap_delta(na, nb, len(na) // 2, reamostras, rng, grupos))
    if not boots:
        return float("nan"), float("nan")
    b = np.mean(boots, axis=0)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def resumo(espelho, reamostras=1000, semente=0, cenas=None, comparar=(("hibrido", "so_bce"),)):
    """Tabela por (conjunto, tamanho, carga, detector, método): P_E médio ± dp entre sementes, ΔP_E vs HILL
    e o IC 95% do ΔP_E (bootstrap pareado nas imagens, média sobre as sementes)."""
    linhas = _ler_tabela(os.path.join(espelho, "resultados_fase40.csv"))
    if not linhas:
        print("tabela vazia"); return []
    rng = np.random.default_rng(semente)
    grupos = {}
    for l in linhas:
        grupos.setdefault((l["conjunto"], int(l["tamanho"]), l["bpp"], l["detector"], l["metodo"]), []).append(l)
    saida = []
    print("=" * 104)
    print("FASE 4.0 — P_E no teste (0,5 = indetectável). ΔP_E > 0 = mais seguro que o HILL")
    print("=" * 104)
    print(f"{'conj':>5}{'tam':>5}{'bpp':>6}{'detector':>11}{'método':>10}{'n':>3}{'P_E':>9}{'dp':>7}"
          f"{'ΔP_E':>9}{'IC 95%':>18}{'PSNR':>8}{'SSIM':>9}{'taxa':>8}")
    for chave in sorted(grupos, key=lambda c: (c[0], c[1], -float(c[2]), c[3], c[4] != "HILL", c[4])):
        conj, tam, bpp, det, met = chave
        g = grupos[chave]
        pes = np.array([float(l["pe"]) for l in g])
        ref = {int(l["semente"]): l for l in grupos.get((conj, tam, bpp, det, "HILL"), [])}
        delta, ic = float("nan"), (float("nan"), float("nan"))
        if met != "HILL":
            pares = [(l, ref[int(l["semente"])]) for l in g if int(l["semente"]) in ref]
            if pares:
                delta = float(np.mean([float(a["pe"]) - float(h["pe"]) for a, h in pares]))
                boots = []
                for a, h in pares:
                    na = _carregar_notas(espelho, met, bpp, det, a["semente"])
                    nh = _carregar_notas(espelho, "HILL", bpp, det, a["semente"])
                    if na is not None and nh is not None:
                        boots.append(_bootstrap_delta(na, nh, int(a["n_teste"]), reamostras, rng, cenas))
                if boots:
                    b = np.mean(boots, axis=0)
                    ic = (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))
        l0 = g[0]
        dp = float(pes.std(ddof=1)) if len(pes) > 1 else float("nan")
        r = {"conjunto": conj, "tamanho": tam, "bpp": bpp, "detector": det, "metodo": met,
             "n_sementes": len(pes), "pe_media": float(pes.mean()), "pe_dp": dp, "delta_pe": delta,
             "ic_baixo": ic[0], "ic_alto": ic[1], "psnr": float(l0["psnr"]),
             "ssim": float(l0["ssim"]), "taxa": float(l0["taxa"])}
        saida.append(r)
        ic_txt = "" if met == "HILL" or ic[0] != ic[0] else f"[{ic[0]:+.3f}, {ic[1]:+.3f}]"
        d_txt = "" if met == "HILL" else f"{delta:+.4f}"
        dp_txt = "" if dp != dp else f"{dp:.4f}"
        print(f"{conj:>5}{tam:>5}{float(bpp):>6.2f}{det:>11}{met:>10}{len(pes):>3}{pes.mean():>9.4f}"
              f"{dp_txt:>7}{d_txt:>9}{ic_txt:>18}{r['psnr']:>8.2f}{r['ssim']:>9.5f}{r['taxa']:>8.2%}")
    print("=" * 104)
    print("Leitura: diferença só conta se o IC 95% do ΔP_E não cruzar 0 (e, de preferência, nos dois detectores).")
    print("Bootstrap:", "por CENA" if cenas is not None else "por imagem")
    for a, b in comparar:
        celulas = sorted({k[:4] for k in grupos if k[4] == a and k[:4] + (b,) in grupos},
                         key=lambda c: (c[0], c[1], -float(c[2]), c[3]))
        if not celulas:
            continue
        print(f"\n{a} − {b} (IC 95% pareado direto; > 0 = {a} mais seguro)")
        for conj, tam, bpp, det in celulas:
            la = {int(l["semente"]): float(l["pe"]) for l in grupos[(conj, tam, bpp, det, a)]}
            lb = {int(l["semente"]): float(l["pe"]) for l in grupos[(conj, tam, bpp, det, b)]}
            sem = sorted(set(la) & set(lb))
            d = float(np.mean([la[s] - lb[s] for s in sem]))
            ic = _ic_pareado(espelho, a, b, bpp, det, sem, reamostras, rng, cenas)
            print(f"{conj:>5}{tam:>5}{float(bpp):>6.2f}{det:>11}  n={len(sem)}  {d:+.4f}  [{ic[0]:+.3f}, {ic[1]:+.3f}]")
            saida.append({"conjunto": conj, "tamanho": tam, "bpp": bpp, "detector": det,
                          "metodo": f"{a}-{b}", "n_sementes": len(sem), "pe_media": float("nan"),
                          "pe_dp": float("nan"), "delta_pe": d, "ic_baixo": ic[0], "ic_alto": ic[1],
                          "psnr": float("nan"), "ssim": float("nan"), "taxa": float("nan")})
    return saida


def _carregar_notas(espelho, metodo, bpp, det, semente):
    c = os.path.join(espelho, nome_notas(metodo, float(bpp), det, int(semente)))
    return np.load(c) if os.path.exists(c) else None
