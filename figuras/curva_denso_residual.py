
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = pathlib.Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "figuras" / "curva_denso_residual.png"

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA2 = "#52514e"
DISCRETO = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"
DENSO = "#2a78d6"
RESIDUAL = "#eb6834"
ALVO = "#008300"

JANELA_SUAVIZACAO = 5
PSNR_ALVO = 35.0


def ler_metricas(caminho):
    with open(caminho, encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    return ([int(l["epoca"]) for l in linhas],
            [float(l["psnr"]) for l in linhas],
            [float(l["ssim"]) for l in linhas])


def montar_trajetoria(csv_fase0, csv_fase2):
  
    epocas0, psnr0, ssim0 = ler_metricas(csv_fase0)
    epocas2, psnr2, ssim2 = ler_metricas(csv_fase2)
    corte = len(epocas0)
    x = epocas0 + [e + corte for e in epocas2]
    return x, psnr0 + psnr2, ssim0 + ssim2, corte


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
    eixo.axvspan(corte + 0.5, fim + 0.5, color=DISCRETO, alpha=0.06, zorder=0)
    eixo.axvline(corte + 0.5, color=DISCRETO, linewidth=1.0,
                 linestyle=(0, (4, 3)), zorder=1)


def desenhar_psnr(eixo, series, corte, fim):
    for x, psnr, cor, rotulo in series:
        eixo.plot(x, psnr, color=cor, alpha=0.20, linewidth=0.9, zorder=2)
        eixo.plot(x, suavizar(psnr), color=cor, linewidth=2.0,
                  label=f"gerador {rotulo}", zorder=3)

    eixo.plot([0, fim], [PSNR_ALVO, PSNR_ALVO], color=ALVO, linewidth=1.1,
              linestyle=(0, (5, 3)), zorder=1)
    eixo.text(3, PSNR_ALVO + 0.6, f"alvo {PSNR_ALVO:.0f} dB".replace(".", ","),
              color=ALVO, fontsize=7.5, va="bottom", style="italic")

    for (x, psnr, cor, rotulo), desloca in zip(reversed(series), (7, -7)):
        eixo.annotate(f"{rotulo} {psnr[-1]:.1f} dB".replace(".", ","),
                      xy=(fim, suavizar(psnr)[-1]), xytext=(6, desloca),
                      textcoords="offset points", color=cor, fontsize=7.5,
                      va="center", fontweight="bold")

    eixo.text(corte + (fim - corte) / 2, 19.0, "WGAN-GP ligada", color=TINTA2,
              fontsize=8, ha="center", va="center", style="italic")
    eixo.set_ylabel("PSNR (dB)", color=TINTA2, fontsize=9)
    eixo.set_ylim(14, 41)
    eixo.set_xlim(0, fim + 34)
    eixo.legend(loc="lower left", bbox_to_anchor=(0.02, -0.02), fontsize=8,
                frameon=False, labelcolor=TINTA, handlelength=1.6, ncol=2,
                columnspacing=1.2)
    eixo.set_title("BOSSBase · 0,0625 bpp - gerador denso × residual, Fases 0 e 2",
                   color=TINTA, fontsize=10.5, fontweight="bold", loc="left", pad=8)


def desenhar_ssim(eixo, series, fim):
    for x, ssim, cor in series:
        eixo.plot(x, suavizar(ssim), color=cor, linewidth=2.0, zorder=3)
    eixo.set_ylabel("SSIM", color=TINTA2, fontsize=9)
    eixo.set_xlabel("época  (Fase 0: 1-120  ·  Fase 2: 121-180)",
                    color=TINTA2, fontsize=9)
    eixo.set_ylim(0.25, 1.0)


def main():
    x_denso, psnr_denso, ssim_denso, corte = montar_trajetoria(
        RAIZ / "runs" / "phase0_bossbase" / "metricas_val.csv",
        RAIZ / "runs" / "phase2_bossbase" / "metricas_val.csv")
    x_resid, psnr_resid, ssim_resid, _ = montar_trajetoria(
        RAIZ / "runs" / "phase0_residual" / "metricas_val.csv",
        RAIZ / "runs" / "phase2_residual" / "metricas_val.csv")
    fim = max(x_denso)

    figura, (painel_psnr, painel_ssim) = plt.subplots(
        2, 1, figsize=(6.8, 3.9), sharex=True, dpi=150,
        gridspec_kw={"height_ratios": [1.5, 1]})
    figura.patch.set_facecolor(SUPERFICIE)
    for eixo in (painel_psnr, painel_ssim):
        preparar_eixo(eixo, corte, fim)

    desenhar_psnr(painel_psnr,
                  [(x_denso, psnr_denso, DENSO, "denso"),
                   (x_resid, psnr_resid, RESIDUAL, "residual")], corte, fim)
    desenhar_ssim(painel_ssim,
                  [(x_denso, ssim_denso, DENSO),
                   (x_resid, ssim_resid, RESIDUAL)], fim)

    figura.tight_layout()
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(SAIDA, facecolor=SUPERFICIE, bbox_inches="tight")
    print("gerado:", SAIDA)

    for rotulo, psnr, ssim in (("denso", psnr_denso, ssim_denso),
                               ("residual", psnr_resid, ssim_resid)):
        print(f"  {rotulo:9} Fase 2: melhor PSNR {max(psnr[corte:]):.2f} dB · "
              f"média 10 últimas {sum(psnr[-10:]) / 10:.2f} dB · "
              f"SSIM {sum(ssim[-10:]) / 10:.4f}")


if __name__ == "__main__":
    main()
