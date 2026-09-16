# Esteganografia em imagens digitais com redes adversárias generativas

Iniciação científica - IFSP, Câmpus Presidente Epitácio.
Breno W. B. Correia (discente) · Kleber M. Trevisani (orientador).

O trabalho constrói um modelo de esteganografia por GAN em camadas: cada fase acrescenta
um componente sobre uma base que já funciona, com critérios de saída declarados antes do
treino. Assim, qualquer instabilidade de convergência tem causa única e identificável, e cada
resultado pode ser atribuído ao componente que o produziu.

O plano completo, com o registro de cada run e as decisões tomadas, está no **[ROADMAP.md](ROADMAP.md)**.

## Resultados até aqui

BOSSBase 1.01, tons de cinza, 128 × 128, carga de 1.024 bits por imagem (0,0625 bpp).

| Fase | O que entra | Acurácia | PSNR | SSIM | L∞ | Épocas ≥ 35 dB |
|---|---|---|---|---|---|---|
| 0/1 | codificador + decodificador, sem adversário | 0,9999 | 35,16 dB | 0,963 | 0,362 | 1 de 120 |
| 2 | crítico WGAN-GP | 0,9997 | 36,30 dB | 0,974 | 0,388 | 56 de 60 |
| 2.5 | gerador residual (STEGS2GAN) | 0,9998 | **38,27 dB** | **0,980** | **0,203** | **60 de 60** |
| 3 | esteganalisador no laço | — | — | — | — | 1º run reprovado |

A Fase 2.5 é a melhor configuração medida. O ganho não é um pico: as 60 épocas do gerador
residual ficam acima de 36,30 dB, que era o melhor ponto isolado do gerador denso. 

## Estrutura

```
modelo/          o código do modelo — fonte única da verdade
  fase0_codificador_decodificador.py   Fases 0/1: gerador denso, treino, métricas, payload
  fase2_wgan_gp.py                     Fase 2: crítico WGAN-GP, penalidade de gradiente
  fase25_residual.py                   Fase 2.5: troca o gerador por blocos residuais
  fase3_esteganalisador.py             Fase 3: esteganalisador no laço adversarial

notebooks/       execução no Colab GPU (autossuficientes: escrevem os .py no ambiente)
experimentos/    rodadas auxiliares em CPU
figuras/         scripts que geram as curvas + os diagramas de arquitetura (PNG)
resumo/          resumo expandido MC²T 2026, gerado a partir do gabarito do evento
ferramentas/     conversão de logs de treino em CSV
datasets/        symlink para ../IC_stego/datasets (não versionado)
runs/            registro dos treinos: logs, CSVs de métricas e checkpoints
```

## Referência

Este trabalho dá continuidade ao **STEGS2GAN** (Souza; Correia; Trevisani, 2026), do qual
incorpora os blocos residuais no gerador (Fase 2.5) e ao qual acrescenta o treinamento por
Wasserstein — ali listado como proposta futura — e a avaliação com esteganalisador
independente como etapa obrigatória.
