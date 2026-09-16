# -*- coding: utf-8 -*-

import pathlib
import sys

RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "modelo"))

import fase0_codificador_decodificador as fase0


CONFIG_LOCAL = dict(
    canais=1,
    tamanho_img=64,          # recorte pequeno: na CPU, 128 fica lento demais
    tamanho_msg=16,          # payload baixo (1/16 bpp) -> ajuda PSNR e acurácia
    max_imagens=2000,        # subconjunto; suba se tiver paciência
    epocas=20,
    largura=32,
    tamanho_lote=16,
    num_trabalhadores=4,
    lambda_imagem=2.0,
    lambda_residuo=0.05,
    salvar_cada=5,
)


def main():
    fase0.CONFIG.update(CONFIG_LOCAL)
    # troque por DIV2K_train_HR ou val2017 (nesses, canais=3)
    fase0.CONFIG["pasta_dados"] = str(RAIZ / "datasets" / "BOSSbase_1.01")
    fase0.CONFIG["pasta_saida"] = str(RAIZ / "runs" / "phase0_local")
    return fase0.treinar()


if __name__ == "__main__":
    main()
