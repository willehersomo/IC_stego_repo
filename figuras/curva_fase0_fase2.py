# -*- coding: utf-8 -*-

import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = pathlib.Path(__file__).resolve().parents[1]
ENTRADA_FASE0 = RAIZ / "runs" / "phase0_bossbase" / "metricas_val.csv"
ENTRADA_FASE2 = RAIZ / "runs" / "phase2_bossbase" / "metricas_val.csv"
SAIDA = RAIZ / "runs" / "phase2_bossbase" / "curva_fase0_fase2.png"

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA2 = "#52514e"
DISCRETO = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"
FASE0 = "#2a78d6"
FASE2 = "#eb6834"
ALVO = "#0ca30c"

PSNR_ALVO = 35.0
JANELA_SUAVIZACAO = 5


def ler_metricas(caminho):
    with open(caminho, encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    return ([int(l["epoca"]) for l in linhas],
            [float(l["psnr"]) for l in linhas],
            [float(l["ssim"]) for l in linhas])


def suavizar(valores, janela=JANELA_SUAVIZACAO):
    suave = []
    for i in range(len(valores)):
        inicio = max(0, i - janela + 1)
        suave.append(sum(valores[inicio:i + 1]) / (i - inicio + 1))
    return suave


def preparar_eixo(eixo, corte, fim):
    eixo.set_facecolor(SUPERFICIE)
    for lado in ("top", "right"):
        eixo.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        eixo.spines[lado].set_color(EIXO)
    eixo.tick_params(colors=DISCRETO, labelsize=8)
    eixo.grid(True, color=GRADE, linewidth=0.6, zorder=0)
    eixo.set_axisbelow(True)
    eixo.axvspan(corte + 0.5, fim + 0.5, color=FASE2, alpha=0.05, zorder=0)
    eixo.axvline(corte + 0.5, color=DISCRETO, linewidth=1.0,
                 linestyle=(0, (4, 3)), zorder=1)


def main():
    epocas0, psnr0, ssim0 = ler_metricas(ENTRADA_FASE0)
    epocas2, psnr2, ssim2 = ler_metricas(ENTRADA_FASE2)
    corte = len(epocas0)
    x2 = [e + corte for e in epocas2]
    fim = max(x2)

    figura, (painel_psnr, painel_ssim) = plt.subplots(
        2, 1, figsize=(6.8, 5.0), sharex=True, dpi=150,
        gridspec_kw={"height_ratios": [1.4, 1]})
    figura.patch.set_facecolor(SUPERFICIE)
    for eixo in (painel_psnr, painel_ssim):
        preparar_eixo(eixo, corte, fim)

    for x, psnr, cor, nome in ((epocas0, psnr0, FASE0, "Fase 0 (base)"),
                               (x2, psnr2, FASE2, "Fase 2 (WGAN-GP)")):
        painel_psnr.plot(x, psnr, color=cor, alpha=0.22, linewidth=0.9)
        painel_psnr.plot(x, suavizar(psnr), color=cor, linewidth=2.0, label=nome)
    painel_psnr.axhline(PSNR_ALVO, color=ALVO, linewidth=1.2,
                        linestyle=(0, (5, 3)), zorder=1)
    painel_psnr.text(2, PSNR_ALVO + 0.2, "alvo 35 dB", color=ALVO, fontsize=7.5,
                     va="bottom", style="italic")
    painel_psnr.text(corte + len(epocas2) / 2, 24, "WGAN-GP\nligada", color=FASE2,
                     fontsize=8.5, ha="center", va="center", fontweight="bold")
    painel_psnr.set_ylabel("PSNR (dB)", color=TINTA2, fontsize=9)
    painel_psnr.set_ylim(15, 37.5)
    painel_psnr.legend(loc="upper left", bbox_to_anchor=(0.01, 0.82), fontsize=8,
                       frameon=False, labelcolor=TINTA)
    painel_psnr.set_title(
        "BOSSBase · 0,0625 bpp — Fase 0 → Fase 2 (linha do tempo contínua)",
        color=TINTA, fontsize=10.5, fontweight="bold", loc="left", pad=8)

    painel_ssim.plot(epocas0, suavizar(ssim0), color=FASE0, linewidth=2.0)
    painel_ssim.plot(x2, suavizar(ssim2), color=FASE2, linewidth=2.0)
    painel_ssim.set_ylabel("SSIM", color=TINTA2, fontsize=9)
    painel_ssim.set_xlabel(f"época (Fase 0: 1–{corte}, Fase 2: {corte + 1}–{fim})",
                           color=TINTA2, fontsize=9)
    painel_ssim.set_ylim(0.25, 1.0)

    figura.tight_layout()
    figura.savefig(SAIDA, facecolor=SUPERFICIE, bbox_inches="tight")
    print("gerado:", SAIDA)
    print(f"  Fase 0: platô PSNR ~{sum(psnr0[-20:]) / 20:.2f} dB · melhor {max(psnr0):.2f}")
    print(f"  Fase 2: platô PSNR ~{sum(psnr2[-20:]) / 20:.2f} dB · melhor {max(psnr2):.2f}")


if __name__ == "__main__":
    main()
