
import os
import shutil
import sys
import time

import torch
import torch.nn as nn

import fase0_codificador_decodificador as base
import fase2_wgan_gp as fase2


class BlocoResidual(nn.Module):

    def __init__(self, canais):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(canais, canais, 3, padding=1),
            nn.BatchNorm2d(canais),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(canais, canais, 3, padding=1),
            nn.BatchNorm2d(canais),
        )

    def forward(self, x):
        return x + self.conv(x)


class CodificadorResidual(nn.Module):
  

    def __init__(self, canais, profundidade_dados, largura=32, blocos=3):
        super().__init__()
        self.features = base.bloco_conv(canais, largura)
        self.entrada = base.bloco_conv(largura + profundidade_dados, largura)
        self.residuais = nn.Sequential(*[BlocoResidual(largura) for _ in range(blocos)])
        self.para_residuo = nn.Conv2d(largura, canais, 3, padding=1)

    def forward(self, portadora, mensagem):
        x = self.features(portadora)
        x = self.entrada(torch.cat([x, mensagem], dim=1))
        x = self.residuais(x)
        residuo = self.para_residuo(x)
        return (portadora + residuo).clamp(0.0, 1.0)


class DecodificadorResidual(nn.Module):

    def __init__(self, canais, profundidade_dados, largura=32, blocos=3):
        super().__init__()
        self.entrada = base.bloco_conv(canais, largura)
        self.residuais = nn.Sequential(*[BlocoResidual(largura) for _ in range(blocos)])
        self.para_logits = nn.Conv2d(largura, profundidade_dados, 3, padding=1)

    def forward(self, estego):
        x = self.entrada(estego)
        x = self.residuais(x)
        return self.para_logits(x)


_ORIGINAIS = {"Codificador": base.Codificador, "Decodificador": base.Decodificador}
_ATUAL = "denso"

BLOCOS_RESIDUAIS = 3   # o STEGS2GAN usa 3


def usar(arquitetura):

    global _ATUAL
    if arquitetura not in ("denso", "residual"):
        raise ValueError('arquitetura deve ser "denso" ou "residual"')

    if arquitetura == "denso":
        base.Codificador = _ORIGINAIS["Codificador"]
        base.Decodificador = _ORIGINAIS["Decodificador"]
    else:
        def _cod(canais, profundidade, largura=32):
            return CodificadorResidual(canais, profundidade, largura, BLOCOS_RESIDUAIS)

        def _dec(canais, profundidade, largura=32):
            return DecodificadorResidual(canais, profundidade, largura, BLOCOS_RESIDUAIS)

        base.Codificador = _cod
        base.Decodificador = _dec

    _ATUAL = arquitetura
    print(f"[arquitetura] gerador = {arquitetura}")
    return arquitetura


def contar_parametros(canais=1, profundidade=1, largura=64):
 
    resultado = {}
    for nome, (cod, dec) in {
        "denso": (_ORIGINAIS["Codificador"], _ORIGINAIS["Decodificador"]),
        "residual": (
            lambda c, p, l: CodificadorResidual(c, p, l, BLOCOS_RESIDUAIS),
            lambda c, p, l: DecodificadorResidual(c, p, l, BLOCOS_RESIDUAIS),
        ),
    }.items():
        c = cod(canais, profundidade, largura)
        d = dec(canais, profundidade, largura)
        resultado[nome] = {
            "codificador": sum(p.numel() for p in c.parameters()),
            "decodificador": sum(p.numel() for p in d.parameters()),
        }
        resultado[nome]["total"] = (resultado[nome]["codificador"]
                                    + resultado[nome]["decodificador"])
    return resultado


class _Tee:

    def __init__(self, caminho):
        self.caminho = caminho
        self.arquivo = None
        self.stdout = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.caminho) or ".", exist_ok=True)
        self.arquivo = open(self.caminho, "a", encoding="utf-8")
        self.stdout = sys.stdout
        sys.stdout = self
        return self

    def write(self, texto):
        self.stdout.write(texto)
        self.arquivo.write(texto)
        self.arquivo.flush()

    def flush(self):
        self.stdout.flush()
        self.arquivo.flush()

    def __exit__(self, *exc):
        sys.stdout = self.stdout
        self.arquivo.close()


def _gerar_csvs(log, pasta):
    try:
        from logs_para_csv import ler_log, gravar
    except ImportError:
        print("[csv] logs_para_csv.py não encontrado; CSVs não gerados")
        return
    treino, val = ler_log(log)
    for linhas, nome in ((treino, "metricas_treino.csv"), (val, "metricas_val.csv")):
        if linhas:
            gravar(linhas, os.path.join(pasta, nome))
            print(f"[csv] {nome}: {len(linhas)} linhas")


def espelhar(pasta, destino):
    if not destino or os.path.abspath(destino) == os.path.abspath(pasta):
        return
    os.makedirs(destino, exist_ok=True)
    copiados = 0
    for nome in os.listdir(pasta):
        origem = os.path.join(pasta, nome)
        if os.path.isfile(origem):
            shutil.copy2(origem, os.path.join(destino, nome))
            copiados += 1
    print(f"[espelho] {copiados} arquivo(s) copiado(s) para {destino}")


def _rodar(modulo, cfg_extra, pasta_saida, espelho, rotulo):
    rotulo = f"{rotulo} ({_ATUAL})"     # o braço denso também passa por aqui, na sanidade
    modulo.CONFIG.update(base.COLAB_GPU)
    modulo.CONFIG.update(cfg_extra)
    modulo.CONFIG["pasta_saida"] = pasta_saida
    os.makedirs(pasta_saida, exist_ok=True)
    log = os.path.join(pasta_saida, "treino.log")

    inicio = time.time()
    with _Tee(log):
        print(f"\n===== {rotulo} iniciado em "
              f"{time.strftime('%Y-%m-%d %H:%M:%S')} =====")
        print(f"[arquitetura] {_ATUAL}  |  blocos residuais = {BLOCOS_RESIDUAIS}")
        print(f"[cfg] " + "  ".join(
            f"{k}={modulo.CONFIG[k]}" for k in
            ("tamanho_img", "tamanho_msg", "tamanho_lote", "largura", "epocas",
             "lambda_imagem", "lambda_residuo") if k in modulo.CONFIG))
        resultado = modulo.treinar()
        minutos = (time.time() - inicio) / 60
        print(f"[tempo ] {rotulo}: {minutos:.1f} min "
              f"({minutos * 60 / max(modulo.CONFIG['epocas'], 1):.1f} s/época)")

    _gerar_csvs(log, pasta_saida)
    espelhar(pasta_saida, espelho)
    return resultado


def treinar_fase0(pasta_saida, espelho=None, **cfg_extra):
    return _rodar(base, cfg_extra, pasta_saida, espelho, "Fase 0")


def treinar_fase2(pasta_saida, espelho=None, **cfg_extra):
    return _rodar(fase2, cfg_extra, pasta_saida, espelho, "Fase 2")


ALVO_ACURACIA = 0.99
ALVO_PSNR_FASE2 = 36.30      # o que a Fase 2 densa entregou; não pode regredir


def _ler_csv(caminho):
    import csv
    if not os.path.exists(caminho):
        return []
    with open(caminho, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def comparar(csv_denso, csv_residual, alvo_psnr=ALVO_PSNR_FASE2, rotulo="Fase 2"):
 
    d, r = _ler_csv(csv_denso), _ler_csv(csv_residual)
    if not r:
        print(f"[comparar] {csv_residual} não encontrado — rode o treino antes.")
        return None
    if not d:
        print(f"[comparar] {csv_denso} não encontrado — sem linha de base para comparar.")
        return None

    n = min(len(d), len(r))
    print(f"\n{'=' * 74}\n{rotulo}: DENSO × RESIDUAL nas primeiras {n} épocas\n{'=' * 74}")

    def resumo(linhas, nome):
        linhas = linhas[:n]
        psnr = [float(x["psnr"]) for x in linhas if x.get("psnr")]
        acc = [float(x["acuracia_bits"]) for x in linhas if x.get("acuracia_bits")]
        ssim = [float(x["ssim"]) for x in linhas if x.get("ssim")]
        melhor = max(psnr) if psnr else 0.0
        ult10 = psnr[-10:] or [0.0]
        print(f"  {nome:<9} melhor PSNR {melhor:6.2f} dB | "
              f"média 10 últimas {sum(ult10) / len(ult10):6.2f} dB | "
              f"acurácia mín {min(acc) if acc else 0:.4f} | "
              f"SSIM máx {max(ssim) if ssim else 0:.4f} | "
              f"épocas >= {alvo_psnr:.2f} dB: {sum(1 for p in psnr if p >= alvo_psnr)}")
        return {"melhor": melhor, "acc_min": min(acc) if acc else 0.0,
                "media10": sum(ult10) / len(ult10)}

    rd, rr = resumo(d, "denso"), resumo(r, "residual")

    print(f"\n  diferença (residual - denso): "
          f"{rr['melhor'] - rd['melhor']:+.2f} dB no melhor, "
          f"{rr['media10'] - rd['media10']:+.2f} dB na média das 10 últimas")

    print(f"\n{'-' * 74}\nCRITÉRIOS DE SAÍDA DA FASE 2.5")
    c1 = rr["acc_min"] >= ALVO_ACURACIA
    c2 = rr["melhor"] >= alvo_psnr
    print(f"  (1) acurácia de bits >= {ALVO_ACURACIA}      -> {rr['acc_min']:.4f}  "
          f"{'OK' if c1 else 'FALHOU'}")
    print(f"  (2) PSNR >= {alvo_psnr:.2f} dB (não regredir) -> {rr['melhor']:.2f}  "
          f"{'OK' if c2 else 'FALHOU'}")
    print("  (3) custo por época medido            -> ver a linha [tempo ] em cada treino.log")
    print("  (4) os dois geradores preservados     -> confira o espelho no Drive")

    if c1 and c2:
        print("\n  Residual passa. Decisão de qual segue: maior PSNR a igual acurácia;")
        print("  empate técnico (< 0,2 dB) fica com o mais barato por época.")
    else:
        print("\n  Residual NÃO bateu os critérios. Resultado negativo também é resultado:")
        print("  registre no ROADMAP e siga com o denso.")
    return {"denso": rd, "residual": rr}
