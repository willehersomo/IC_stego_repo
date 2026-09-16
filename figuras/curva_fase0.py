# -*- coding: utf-8 -*-

import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = pathlib.Path(__file__).resolve().parents[1]
ENTRADA = RAIZ / "runs" / "phase0_bossbase" / "metricas_val.csv"
SAIDA = RAIZ / "runs" / "phase0_bossbase" / "curva_psnr.png"

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA2 = "#52514e"
DISCRETO = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"
AZUL = "#2a78d6"
VERDE = "#008300"
ALVO = "#0ca30c"

PSNR_ALVO = 35.0
ACURACIA_ALVO = 0.99
JANELA_SUAVIZACAO = 5


def ler_metricas(caminho):
    with open(caminho, encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    return ([int(l["epoca"]) for l in linhas],
            [float(l["acuracia_bits"]) for l in linhas],
            [float(l["psnr"]) for l in linhas],
            [float(l["ssim"]) for l in linhas])


def suavizar(valores, janela=JANELA_SUAVIZACAO):
    suave = []
    for i in range(len(valores)):
        inicio = max(0, i - janela + 1)
        suave.append(sum(valores[inicio:i + 1]) / (i - inicio + 1))
    return suave


def indice_melhor(acuracia, psnr):
    """Mesma regra do treino: entre as épocas que batem a acurácia, a de maior PSNR."""
    return max(range(len(psnr)),
               key=lambda i: (acuracia[i] >= ACURACIA_ALVO, psnr[i]))


def preparar_eixo(eixo):
    eixo.set_facecolor(SUPERFICIE)
    for lado in ("top", "right"):
        eixo.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        eixo.spines[lado].set_color(EIXO)
    eixo.tick_params(colors=DISCRETO, labelsize=8)
    eixo.grid(True, color=GRADE, linewidth=0.6, zorder=0)
    eixo.set_axisbelow(True)


def desenhar_psnr(eixo, epocas, psnr, melhor):
    eixo.plot(epocas, psnr, color=AZUL, alpha=0.28, linewidth=1.0, zorder=2)
    eixo.plot(epocas, suavizar(psnr), color=AZUL, linewidth=2.0, zorder=3)
    eixo.axhline(PSNR_ALVO, color=ALVO, linewidth=1.3, linestyle=(0, (5, 3)), zorder=1)
    eixo.text(2, PSNR_ALVO + 0.35, "alvo 35 dB", color=ALVO, fontsize=7.5,
              va="bottom", style="italic")
    eixo.scatter([epocas[melhor]], [psnr[melhor]], s=42, color=ALVO, zorder=5,
                 edgecolor=SUPERFICIE, linewidth=1.2)
    eixo.annotate(f"melhor: época {epocas[melhor]}\n{psnr[melhor]:.2f} dB".replace(".", ","),
                  (epocas[melhor], psnr[melhor]), xytext=(-8, -34),
                  textcoords="offset points", fontsize=7.5, color=TINTA, ha="right",
                  arrowprops=dict(arrowstyle="-", color=DISCRETO, linewidth=0.8))
    eixo.set_ylabel("PSNR (dB)", color=TINTA2, fontsize=9)
    eixo.set_ylim(15, 37)
    eixo.set_title("Fase 0/1 · BOSSBase · 0,0625 bpp — evolução por época",
                   color=TINTA, fontsize=10.5, fontweight="bold", loc="left", pad=8)


def desenhar_ssim_e_acuracia(eixo, epocas, acuracia, ssim):
    eixo.plot(epocas, ssim, color=VERDE, linewidth=2.0, zorder=3, label="SSIM")
    eixo.plot(epocas, acuracia, color=AZUL, linewidth=2.0, zorder=3,
              label="acurácia de bits")
    eixo.set_ylabel("SSIM / acurácia", color=TINTA2, fontsize=9)
    eixo.set_xlabel("época", color=TINTA2, fontsize=9)
    eixo.set_ylim(0.2, 1.03)
    eixo.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=TINTA)
    # rótulos diretos dos valores finais
    eixo.text(epocas[-1] + 1.5, ssim[-1], f"{ssim[-1]:.3f}".replace(".", ","),
              color=VERDE, fontsize=7.5, va="center")
    eixo.text(epocas[-1] + 1.5, acuracia[-1], f"{acuracia[-1]:.3f}".replace(".", ","),
              color=AZUL, fontsize=7.5, va="center")


def main():
    epocas, acuracia, psnr, ssim = ler_metricas(ENTRADA)
    melhor = indice_melhor(acuracia, psnr)

    figura, (painel_psnr, painel_qualidade) = plt.subplots(
        2, 1, figsize=(6.6, 5.0), sharex=True, dpi=150,
        gridspec_kw={"height_ratios": [1.25, 1]})
    figura.patch.set_facecolor(SUPERFICIE)
    for eixo in (painel_psnr, painel_qualidade):
        preparar_eixo(eixo)

    desenhar_psnr(painel_psnr, epocas, psnr, melhor)
    desenhar_ssim_e_acuracia(painel_qualidade, epocas, acuracia, ssim)

    figura.tight_layout()
    figura.savefig(SAIDA, facecolor=SUPERFICIE, bbox_inches="tight")
    print("gerado:", SAIDA)
    print(f"  épocas={len(epocas)}  melhor=época {epocas[melhor]}  "
          f"psnr={psnr[melhor]:.2f}  ssim={ssim[melhor]:.4f}  acc={acuracia[melhor]:.4f}")
    print(f"  épocas com PSNR >= {PSNR_ALVO:.0f}: "
          f"{[e for e, p in zip(epocas, psnr) if p >= PSNR_ALVO]}")


if __name__ == "__main__":
    main()
