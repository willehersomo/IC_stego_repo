# -*- coding: utf-8 -*-

import pathlib
import sys

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "modelo"))

import fase0_codificador_decodificador as fase0


LAMBDAS = (1.0, 5.0, 15.0, 40.0)

CONFIG_VARREDURA = dict(
    canais=1,
    tamanho_img=64,
    max_imagens=2000,
    epocas=8,
    tamanho_lote=16,
    num_trabalhadores=2,
    lambda_mensagem=1.0,
    salvar_cada=999,         # sem amostras por época: aqui só interessa o número
)


def rodar_um(lambda_imagem):
    fase0.CONFIG.update(CONFIG_VARREDURA)
    fase0.CONFIG["pasta_dados"] = str(RAIZ / "datasets" / "BOSSbase_1.01")
    fase0.CONFIG["lambda_imagem"] = lambda_imagem
    fase0.CONFIG["pasta_saida"] = str(RAIZ / "runs" / "sweep" / f"lam_img_{lambda_imagem:g}")
    print(f"\n========== lambda_imagem = {lambda_imagem} ==========")
    return fase0.treinar()


def imprimir_resumo(resultados):
    print("\n\n==================== RESUMO DA VARREDURA ====================")
    print(f"{'lambda_imagem':>13} | {'acuracia_bits':>13} | {'PSNR (dB)':>9}")
    print("-" * 43)
    for lambda_imagem, acuracia, psnr in resultados:
        print(f"{lambda_imagem:>13g} | {acuracia:>13.4f} | {psnr:>9.2f}")
    print("\nLeitura: subir lambda_imagem melhora o PSNR (imperceptibilidade), mas")
    print("tende a derrubar a acurácia de bits. A Fase 2 reequilibra esse trade-off.")


def main():
    resultados = []
    for lambda_imagem in LAMBDAS:
        saida = rodar_um(lambda_imagem)
        resultados.append((lambda_imagem, saida["acuracia_bits"], saida["psnr"]))
    imprimir_resumo(resultados)
    return resultados


if __name__ == "__main__":
    main()
