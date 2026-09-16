"""
Fase 3 — Indetectabilidade dirigida: SRNet como 2º adversário.

O que muda em relação à Fase 2:
  - Entra um SEGUNDO adversário, a SRNet (Boroumand, Chen e Fridrich, 2019). Os dois
    adversários cobram coisas diferentes do gerador e por isso convivem:
      * Crítico WGAN-GP -> REALISMO. Score escalar de "quão natural é esta imagem".
      * SRNet           -> INDETECTABILIDADE. Classificador portadora vs. estego, treinado
                           para achar o micro-ruído residual que a ocultação deixa.
    Uma imagem pode ser perfeitamente realista e ainda assim ter assinatura estatística
    detectável — é exatamente essa lacuna que a SRNet fecha.
  - A perda do gerador ganha um 4º termo: enganar a SRNet (fazer a estego ser classificada
    como portadora).

Por que a SRNet e não outro detector:
  - Ela não usa filtro passa-alta fixo: aprende o residual sozinha.
  - As 7 primeiras camadas NÃO têm pooling, de propósito. Pooling é filtro passa-baixa e
    destruiria justamente o sinal de alta frequência que denuncia a esteganografia. A redução
    de resolução só começa na camada 8.

Warm start: gerador e crítico vêm da Fase 2 (ckpt_best). A SRNet começa DO ZERO — um detector
pré-treinado contra outro gerador entra viciado e mede a coisa errada.

CRITÉRIOS DE SAÍDA da Fase 3 (os três precisam valer):
  (1) recuperação:        acuracia_bits(val) >= 0.99   (não regredir o que a Fase 1 conquistou)
  (2) imperceptibilidade: PSNR(val) não cair mais que ~1 dB em relação aos 36,30 dB da Fase 2
  (3) o detector do laço estacionar perto do acaso (AUC ~ 0.5)

>>> LEIA ISTO ANTES DE COMEMORAR O CRITÉRIO (3) <<<
A AUC reportada aqui é a do detector que PARTICIPOU DO TREINO. Ela pode estar em 0,5 porque o
gerador ficou bom OU porque aquela instância de SRNet não convergiu — os dois casos produzem o
mesmo número. Uma AUC de 0,5 no laço NÃO é evidência de segurança; é evidência sobre aquele
adversário específico. A prova real é a Fase 4, que treina um esteganalisador independente e
mede P_E. Ver ROADMAP.md.
E lembre da simetria: AUC abaixo de 0,5 não é "melhor que aleatório". Significa que o detector
aprendeu a separação com os rótulos invertidos, e basta inverter a regra de decisão. Por isso
o log imprime também max(AUC, 1-AUC).

O QUE O GERADOR PEDE À SRNet (cfg["alvo_srnet"]) — importa mais do que parece:
  - "portadora" (alvo 0) é o objetivo clássico de "enganar o classificador". Ele NÃO tem mínimo
    no equilíbrio: a perda decresce monotonicamente conforme o logit vai a -infinito, então o
    ótimo é "mais portadora que as portadoras". O gerador passa do ponto e a AUC despenca ABAIXO
    de 0,5 — as imagens voltam a ser separáveis, agora com o sinal trocado.
  - "equilibrio" (alvo suave 0,5) tem mínimo EXATO em probabilidade 0,5, que é literalmente
    "o detector não consegue decidir". É o default, e é o que a Fase 3 quer.
  Com "equilibrio", a coluna `enganar` passa a ter piso em ln 2 ~ 0,693 em vez de 0: 0,693 é o
  equilíbrio perfeito, e valores acima indicam separação — em qualquer das duas direções.

Reaproveita: Codificador/Decodificador/loaders/métricas da Fase 0; Crítico e penalidade de
gradiente da Fase 2.

Rodar:  python3.10 fase3_esteganalisador.py
"""
import os
import csv
import time
import random
import shutil
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

import fase0_codificador_decodificador as base    # Codificador, Decodificador, loaders, métricas
import fase2_wgan_gp as fase2            # Critico, penalidade_gradiente


# ---------------------------------------------------------------------------
# CONFIG — herda TODAS as chaves da Fase 2 (que já herda as da Fase 0).
# ---------------------------------------------------------------------------
CONFIG = dict(fase2.CONFIG)
CONFIG.update({
    # --- 2º adversário (SRNet) ---
    # Limiar do critério (2). A Fase 2 fechou em 36,30 dB; o ROADMAP admite cair no máximo
    # ~1 dB. Fica em cfg para que a marca por época e o veredito final usem O MESMO número —
    # antes a marca usava 35,0 e o veredito 35,3, e épocas como 35,06/35,22 apareciam
    # aprovadas no log e reprovavam no fim.
    "alvo_psnr": 35.3,

    "lambda_srnet": 0.01,      # peso do termo "enganar a SRNet" no gerador. É O NOVO KNOB.
                               # Mesmo trade-off do lambda_adv: alto demais derruba a acurácia
                               # de bits, baixo demais não muda nada. Comece em 0,01 e suba.
    "n_srnet": 1,              # passos da SRNet por passo do gerador. Custa ~+22% no tempo por
                               # passo (o gargalo é o gerador denso, não a SRNet). Suba para 2
                               # se a srnet_bce ficar travada em ~0,69 (detector não aprendeu).
    "alvo_srnet": "equilibrio", # O QUE o gerador pede à SRNet. Ver nota no topo do arquivo.
                               #  "equilibrio" -> alvo suave 0,5: mínimo exato em "o detector não
                               #                  consegue decidir". É o que a Fase 3 quer.
                               #  "portadora"  -> alvo 0: decresce sem parar, o gerador passa do
                               #                  ponto e a AUC cai ABAIXO de 0,5. Só para comparar.
    "lr_srnet": 1e-4,          # learning rate da SRNet
    "srnet_compacta": False,   # True = metade dos filtros. Use para uma primeira rodada rápida
                               # de sanidade; os resultados finais devem ser com a SRNet cheia.

    # --- treino ---
    "epocas": 60,
    "usar_amp": False,         # a GP do crítico continua no laço; ver docstring da Fase 2
    "carregar_fase0": None,    # superado: o warm start agora vem da Fase 2
    "carregar_fase2": "runs/phase2_bossbase/ckpt_best.pt",
    "pasta_saida": "runs/phase3_bossbase",

    # --- persistência ---
    # Espelha ckpt_best.pt para fora do ambiente efêmero a cada melhoria. No Colab, aponte
    # para o Drive montado. Foi a ausência disto que fez os pesos das Fases 0 e 2 se perderem.
    "espelho_ckpt": None,      # ex.: "/content/drive/MyDrive/IC_stego/phase3"
})

# Preset GPU (Colab): junta o preset da Fase 2 com os ajustes do 2º adversário.
COLAB_GPU = dict(fase2.COLAB_GPU)
COLAB_GPU.update({
    "epocas": 60,
    "lambda_srnet": 0.01,
    "usar_amp": False,
    # Detecta as CPUs da máquina em vez de fixar um número: o T4 do Colab tem 2 vCPU e a L4
    # tem 8. Fixar 4 gera aviso no T4 e desperdiça a L4.
    "num_trabalhadores": max(1, min(4, (os.cpu_count() or 2) - 1)),
})


# ---------------------------------------------------------------------------
# SRNet — esteganalisador profundo (Boroumand, Chen e Fridrich, 2019)
#
# Quatro tipos de camada, nesta ordem:
#   Tipo 1 (x2)  conv-BN-ReLU                       -> extrai o residual, sem reduzir resolução
#   Tipo 2 (x5)  residual: conv-BN-ReLU-conv-BN + atalho
#   Tipo 3 (x4)  residual com redução: mesmo ramo + avg-pool 3x3 stride 2; atalho 1x1 stride 2
#   Tipo 4 (x1)  conv-BN + global average pooling   -> vetor de estatísticas da imagem
# Depois: uma camada linear -> 1 logit (portadora vs. estego).
# ---------------------------------------------------------------------------
class _Tipo1(nn.Module):
    def __init__(self, entrada, saida):
        super().__init__()
        self.conv = nn.Conv2d(entrada, saida, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(saida)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class _Tipo2(nn.Module):
    """Residual sem redução de resolução: saída = x + F(x)."""
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
    """Residual COM redução: ramo principal termina em avg-pool; atalho é conv 1x1 stride 2."""
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
    """Última convolução + global average pooling -> (B, canais)."""
    def __init__(self, entrada, saida):
        super().__init__()
        self.c = nn.Conv2d(entrada, saida, 3, padding=1, bias=False)
        self.b = nn.BatchNorm2d(saida)

    def forward(self, x):
        h = F.relu(self.b(self.c(x)), inplace=True)
        return h.mean(dim=(2, 3))          # média espacial global


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
        h = self.tipo1(x)      # sem pooling: preserva o residual de alta frequência
        h = self.tipo2(h)      # ainda sem pooling
        h = self.tipo3(h)      # a partir daqui, reduz resolução
        h = self.tipo4(h)      # (B, C)
        return self.para_logit(h).squeeze(1)   # (B,)


# ---------------------------------------------------------------------------
# Métrica de detecção
# ---------------------------------------------------------------------------
def auc_binaria(scores, rotulos):
    """AUC pela estatística de Mann-Whitney (equivale à área sob a curva ROC).

    scores  : tensor (N,) — saída contínua do detector
    rotulos : tensor (N,) — 1 = estego, 0 = portadora

    Interpretação: 0,5 = classificador aleatório. Independe do limiar de decisão, e é por
    isso que ela vale mais do que a acurácia para julgar um detector.
    """
    scores = scores.detach().float().flatten()
    rotulos = rotulos.detach().float().flatten()
    n_pos = int(rotulos.sum().item())
    n_neg = int(rotulos.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ordem = torch.argsort(scores)
    postos = torch.empty_like(scores)
    postos[ordem] = torch.arange(1, scores.numel() + 1, dtype=scores.dtype, device=scores.device)
    soma_postos_pos = postos[rotulos == 1].sum().item()
    return (soma_postos_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


@torch.no_grad()
def avaliar_srnet(codificador, srnet, loader, cfg):
    """Roda a SRNet sobre pares (portadora, estego) da validação e devolve AUC e acurácia."""
    dispositivo = cfg["dispositivo"]
    codificador.eval(); srnet.eval()
    todos_scores, todos_rotulos = [], []

    for portadora in loader:
        portadora = portadora.to(dispositivo, non_blocking=True)
        b = portadora.size(0)
        mensagem = base.mensagem_aleatoria(b, cfg)
        estego = codificador(portadora, base.espalhar_mensagem(mensagem, cfg))

        logits = srnet(torch.cat([portadora, estego], dim=0)).cpu()   # lote único, como no treino
        todos_scores.append(logits)
        todos_rotulos.append(torch.cat([torch.zeros(b), torch.ones(b)]))

    scores = torch.cat(todos_scores)
    rotulos = torch.cat(todos_rotulos)
    auc = auc_binaria(scores, rotulos)
    acc = ((scores > 0).float() == rotulos).float().mean().item()
    return {"auc": auc, "acuracia_deteccao": acc}


# ---------------------------------------------------------------------------
# Warm start
# ---------------------------------------------------------------------------
def _carregar_gerador(caminho, codificador, decodificador, critico, dispositivo, log=print):
    """Carrega pesos da Fase 2. Tolera o nome antigo das chaves (encoder/decoder)."""
    if not caminho or not os.path.exists(caminho):
        log(f"[warm start] {caminho!r} não encontrado -> gerador DO ZERO.\n"
              "             A Fase 3 é para ser construída SOBRE a Fase 2. Rode a Fase 2 antes,\n"
              "             ou aponte 'carregar_fase2' para o ckpt_best.pt no Drive.")
        return False

    estado = torch.load(caminho, map_location=dispositivo)
    cod = estado.get("codificador", estado.get("encoder"))
    dec = estado.get("decodificador", estado.get("decoder"))
    if cod is None or dec is None:
        raise KeyError(f"{caminho}: não achei os pesos do codificador/decodificador. "
                       f"Chaves presentes: {list(estado.keys())}")
    codificador.load_state_dict(cod)
    decodificador.load_state_dict(dec)

    if "critico" in estado:
        critico.load_state_dict(estado["critico"])
        log(f"[warm start] gerador + crítico carregados de {caminho} (época {estado.get('epoca', '?')})")
    else:
        log(f"[warm start] gerador carregado de {caminho}; crítico DO ZERO "
              "(checkpoint não tinha os pesos do crítico)")
    return True


class _Log:
    """Imprime na tela E grava em arquivo, com flush a cada linha.

    O flush importa: se a sessão do Colab cair no meio do treino, o que já foi impresso
    continua no disco. Sem ele, o buffer se perde junto com o runtime.
    """
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
    """Copia o checkpoint para fora do ambiente efêmero (Drive), se configurado."""
    destino = cfg.get("espelho_ckpt")
    if not destino:
        return
    # Se o espelho aponta para a própria pasta_saida (caso comum quando se roda direto no
    # Drive montado), copiar é um no-op que só levanta SameFileError a cada época. Silencia
    # sem mascarar falhas reais de escrita.
    if os.path.abspath(destino) == os.path.abspath(cfg["pasta_saida"]):
        return
    try:
        os.makedirs(destino, exist_ok=True)
        shutil.copy2(origem, os.path.join(destino, os.path.basename(origem)))
    except Exception as e:
        log(f"[espelho] falhou ao copiar para {destino}: {e}")


# ---------------------------------------------------------------------------
# Treino com DOIS adversários
# ---------------------------------------------------------------------------
def treinar():
    cfg = CONFIG
    torch.manual_seed(cfg["semente"]); random.seed(cfg["semente"])
    os.makedirs(cfg["pasta_saida"], exist_ok=True)
    dispositivo = cfg["dispositivo"]
    tm = base._tamanho_msg(cfg)
    bpp = cfg["profundidade_dados"] * (tm / cfg["tamanho_img"]) ** 2

    # Tudo que for impresso daqui em diante também vai para treino.log.
    log = _Log(os.path.join(cfg["pasta_saida"], "treino.log"))
    log(f"===== Fase 3 iniciada em {datetime.now():%Y-%m-%d %H:%M:%S} =====")
    log("[cfg] " + "  ".join(f"{k}={cfg[k]}" for k in
        ("tamanho_img", "tamanho_msg", "tamanho_lote", "largura", "epocas",
         "lambda_imagem", "lambda_mensagem", "lambda_residuo", "lambda_adv",
         "lambda_srnet", "n_critic", "n_srnet", "lr_srnet", "srnet_compacta")))
    log(f"[infra] dispositivo={dispositivo}  payload={bpp:.4g} bpp  "
          f"lambda_adv={cfg['lambda_adv']}  lambda_srnet={cfg['lambda_srnet']}  "
          f"n_critic={cfg['n_critic']}  n_srnet={cfg['n_srnet']}"
          + ("  [SRNet COMPACTA]" if cfg["srnet_compacta"] else ""))

    loader_treino, loader_val = base.montar_loaders(cfg)

    codificador = base.Codificador(cfg["canais"], cfg["profundidade_dados"], cfg["largura"]).to(dispositivo)
    decodificador = base.Decodificador(cfg["canais"], cfg["profundidade_dados"], cfg["largura"]).to(dispositivo)
    critico = fase2.Critico(cfg["canais"], cfg["largura_critico"]).to(dispositivo)
    srnet = SRNet(cfg["canais"], compacta=cfg["srnet_compacta"]).to(dispositivo)

    _carregar_gerador(cfg.get("carregar_fase2"), codificador, decodificador, critico,
                      dispositivo, log=log)
    n_par = sum(p.numel() for p in srnet.parameters())
    log(f"[srnet] iniciada DO ZERO — {n_par/1e6:.2f}M parâmetros")

    opt_g = torch.optim.Adam(list(codificador.parameters()) + list(decodificador.parameters()),
                             lr=cfg["taxa_aprend"], betas=(cfg["beta1"], cfg["beta2"]))
    opt_c = torch.optim.Adam(critico.parameters(),
                             lr=cfg["lr_critico"], betas=(cfg["beta1"], cfg["beta2"]))
    opt_s = torch.optim.Adam(srnet.parameters(), lr=cfg["lr_srnet"])   # betas padrão: é classificador, não crítico

    caminho_csv = os.path.join(cfg["pasta_saida"], "metricas_val.csv")
    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoca", "acuracia_bits", "psnr", "ssim", "auc_srnet", "w_dist"])

    # Diagnóstico do laço de treino. São estas colunas que dizem SE o treino está saudável —
    # 'srnet_bce' travada em ~0,69 = detector não aprendeu; 'enganar' crescendo sem parar = o
    # termo da SRNet está dominando a perda do gerador.
    caminho_csv_treino = os.path.join(cfg["pasta_saida"], "metricas_treino.csv")
    with open(caminho_csv_treino, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoca", "perda_c", "w_dist", "adv", "srnet_bce", "enganar",
                                "img_mse", "msg_bce", "acuracia_bits"])

    melhor = {"score": -1.0, "epoca": 0, "acuracia_bits": 0.0, "psnr": 0.0, "ssim": 0.0, "auc": float("nan")}

    t_inicio = time.time()
    for epoca in range(1, cfg["epocas"] + 1):
        t_epoca = time.time()
        codificador.train(); decodificador.train(); critico.train(); srnet.train()
        ac = {"c": 0.0, "w": 0.0, "adv": 0.0, "s": 0.0, "eng": 0.0,
              "img": 0.0, "msg": 0.0, "acc": 0.0, "n": 0}

        for portadora in loader_treino:
            portadora = portadora.to(dispositivo, non_blocking=True)
            b = portadora.size(0)

            # ============ 1) CRÍTICO (realismo), n_critic vezes ============
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

            # ============ 2) SRNet (detecção), n_srnet vezes ============
            # Classificador binário puro: portadora = 0, estego = 1.
            # Lote ÚNICO com portadora e estego juntas. A SRNet tem BatchNorm em todas as
            # camadas; se as duas classes vierem em passagens separadas, cada lote é normalizado
            # pelas próprias estatísticas e a rede pode separar pela composição do lote em vez
            # do conteúdo — um viés que só aparece depois, em eval(). Concatenar elimina isso.
            for _ in range(cfg["n_srnet"]):
                mensagem = base.mensagem_aleatoria(b, cfg)
                with torch.no_grad():
                    estego = codificador(portadora, base.espalhar_mensagem(mensagem, cfg))
                logits_s = srnet(torch.cat([portadora, estego], dim=0))
                alvos_s = torch.cat([torch.zeros(b, device=dispositivo),
                                     torch.ones(b, device=dispositivo)])
                perda_s = F.binary_cross_entropy_with_logits(logits_s, alvos_s)
                opt_s.zero_grad(); perda_s.backward(); opt_s.step()

            # ============ 3) GERADOR, 1 vez, contra os DOIS adversários ============
            mensagem = base.mensagem_aleatoria(b, cfg)
            estego = codificador(portadora, base.espalhar_mensagem(mensagem, cfg))
            logits = base.agregar_logits(decodificador(estego), cfg)

            perda_img = F.mse_loss(estego, portadora)
            perda_msg = F.binary_cross_entropy_with_logits(logits, mensagem)
            perda_res = (estego - portadora).abs().mean()
            perda_adv = -critico(estego).mean()          # quer SUBIR o score de realismo

            # Mesma composição de lote do passo da SRNet, para a BatchNorm ver as mesmas
            # estatísticas; o gradiente só flui pela metade estego.
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
        restam = (cfg["epocas"] - epoca) * (time.time() - t_inicio) / epoca
        log(f"[tempo ] época {epoca:3d}  {seg:6.1f}s  ({ac['n']//cfg['tamanho_lote']} passos de gerador)"
            f"  restam ~{restam/60:.0f} min")
        log(f"[treino] época {epoca:3d}  perda_c={ac['c']/n:+.3f}  W={ac['w']/n:+.3f}  "
              f"srnet_bce={ac['s']/n:.4f}  enganar={ac['eng']/n:.4f}  "
              f"img_mse={ac['img']/n:.5f}  msg_bce={ac['msg']/n:.4f}  "
              f"acuracia_bits={ac['acc']/n:.4f}")

        # --- validação: qualidade (Fase 0) + detecção (nova) ---
        val = base.avaliar(codificador, decodificador, loader_val, cfg)
        det = avaliar_srnet(codificador, srnet, loader_val, cfg)
        auc, auc_ef = det["auc"], max(det["auc"], 1.0 - det["auc"])

        c1 = val["acuracia_bits"] >= cfg["alvo_bitacc"]
        c2 = val["psnr"] >= cfg["alvo_psnr"]
        c3 = abs(auc - 0.5) <= 0.05
        marca = "".join(["1" if c1 else ".", "2" if c2 else ".", "3" if c3 else "."])
        log(f"[  val ] época {epoca:3d}  acuracia_bits={val['acuracia_bits']:.4f}  "
              f"psnr={val['psnr']:.2f}dB  ssim={val['ssim']:.4f}  "
              f"auc_srnet={auc:.4f} (efetiva {auc_ef:.4f})  criterios[{marca}]")

        with open(caminho_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoca, f"{val['acuracia_bits']:.4f}", f"{val['psnr']:.2f}",
                                    f"{val['ssim']:.4f}", f"{auc:.4f}", f"{ac['w']/n:.4f}"])
        with open(caminho_csv_treino, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoca, f"{ac['c']/n:.4f}", f"{ac['w']/n:.4f}",
                                    f"{ac['adv']/n:.4f}", f"{ac['s']/n:.4f}", f"{ac['eng']/n:.4f}",
                                    f"{ac['img']/n:.6f}", f"{ac['msg']/n:.4f}", f"{ac['acc']/n:.4f}"])

        # Espelha os registros TODA época — não só quando o checkpoint melhora. Se a sessão
        # cair na época 47, o log e as métricas até ali já estão salvos fora do runtime.
        for reg in (caminho_csv, caminho_csv_treino, os.path.join(cfg["pasta_saida"], "treino.log")):
            _espelhar(cfg, reg, log)

        # --- melhor checkpoint ---
        # Critério DELIBERADAMENTE igual ao da Fase 2 (PSNR com acurácia como porteira), e não
        # a AUC do laço. Selecionar pela AUC do detector que participou do treino é escolher o
        # checkpoint que melhor engana ESTE detector — exatamente o viés que a Fase 4 existe
        # para não cometer. A AUC fica no log como monitoramento, não como critério.
        score = (val["psnr"] if c1 else val["acuracia_bits"] - 100.0)
        if score > melhor["score"]:
            melhor = {"score": score, "epoca": epoca, "auc": auc,
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
            torch.save({"codificador": codificador.state_dict(),
                        "decodificador": decodificador.state_dict(),
                        "critico": critico.state_dict(),
                        "srnet": srnet.state_dict(),
                        "epoca": epoca, "cfg": cfg},
                       destino)
            _espelhar(cfg, destino, log)

    # --- fechamento ---
    auc_ef = max(melhor["auc"], 1.0 - melhor["auc"])
    log(f"\n[fim] checkpoints, amostras e metricas_val.csv em: {cfg['pasta_saida']}")
    log(f"[melhor] época {melhor['epoca']}: acuracia_bits={melhor['acuracia_bits']:.4f}  "
          f"psnr={melhor['psnr']:.2f}dB  ssim={melhor['ssim']:.4f}  auc_srnet={melhor['auc']:.4f}")
    log("\nCRITÉRIOS DE SAÍDA DA FASE 3:")
    log(f"  (1) acuracia_bits >= 0.99          -> {melhor['acuracia_bits']:.4f}  "
          f"{'OK' if melhor['acuracia_bits'] >= 0.99 else 'FALHOU: baixe lambda_srnet'}")
    log(f"  (2) PSNR >= {cfg['alvo_psnr']} dB (Fase 2 - 1dB) -> {melhor['psnr']:.2f}  "
          f"{'OK' if melhor['psnr'] >= cfg['alvo_psnr'] else 'FALHOU: baixe lambda_srnet ou suba lambda_imagem'}")
    if abs(melhor["auc"] - 0.5) <= 0.05:
        veredito = "OK"
    elif melhor["auc"] > 0.5:
        veredito = "detector separando: suba lambda_srnet"
    else:
        veredito = ("gerador PASSOU DO PONTO (estego com score menor que portadora). "
                    "NÃO suba lambda_srnet — isso piora. Use alvo_srnet='equilibrio'")
    log(f"  (3) AUC do laço ~ 0.5              -> {melhor['auc']:.4f} (efetiva {auc_ef:.4f})  {veredito}")
    log("\nATENÇÃO: o critério (3) mede o detector que participou do treino. Ele NÃO prova")
    log("segurança — pode significar que aquela SRNet não convergiu. A prova é a Fase 4,")
    log("que treina um esteganalisador independente e mede P_E. Ver ROADMAP.md.")
    log(f"\n[registros] treino.log, metricas_val.csv e metricas_treino.csv em {cfg['pasta_saida']}")
    if not cfg.get("espelho_ckpt"):
        log("\nLEMBRETE: 'espelho_ckpt' está vazio. Se isto rodou no Colab, BAIXE AGORA o")
        log("ckpt_best.pt E os arquivos de registro — foi assim que os pesos das Fases 0 e 2")
        log("se perderam, e a saída da célula do notebook some junto com o runtime.")
    else:
        _espelhar(cfg, os.path.join(cfg["pasta_saida"], "treino.log"), log)
        log(f"[espelho] registros e checkpoints copiados para {cfg['espelho_ckpt']}")
    log.fechar()
    return {"acuracia_bits": melhor["acuracia_bits"], "psnr": melhor["psnr"],
            "ssim": melhor["ssim"], "auc": melhor["auc"]}


if __name__ == "__main__":
    treinar()
