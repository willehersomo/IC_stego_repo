# Esteganografia em imagens digitais com redes adversárias generativas

Iniciação científica — IFSP, Câmpus Presidente Epitácio.
Breno W. B. Correia (discente) · Kleber M. Trevisani (orientador).

O trabalho constrói um modelo de esteganografia por GAN **em camadas**: cada fase acrescenta
*um* componente sobre uma base que já funciona, com critérios de saída declarados antes do
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
residual ficam acima de 36,30 dB, que era o *melhor ponto isolado* do gerador denso. O
confundidor a declarar é que o gerador residual tem 2,13× mais parâmetros a igual largura de
camada — a ablação a igual número de parâmetros está prevista na Fase 5.

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

## Como rodar

```bash
pip install -r requisitos.txt

python3 experimentos/treino_local_cpu.py # teste rápido em CPU
python3 figuras/curva_denso_residual.py  # refaz a figura da comparação
python3 resumo/gerar_resumo_docx.py      # refaz o resumo expandido
```

Os treinos que fecham as fases rodam em GPU, pelos notebooks em `notebooks/`. Os scripts de
`figuras/`, `resumo/` e `ferramentas/` resolvem os caminhos a partir da raiz do repositório —
podem ser chamados de qualquer diretório.

## O que não está versionado, e por quê

| Item | Onde está | Motivo |
|---|---|---|
| `datasets/` (6,6 GB) | symlink para `../IC_stego/datasets` | tamanho |
| `*.pt` (~31 MB) | no disco local e no Google Drive | git não lida bem com binários que mudam a cada run |
| `runs/**/samples_epoch*.png` | no disco local | centenas de PNGs por run, regeneráveis |
| `resumo/resumo_mc2t_2026.docx` | gerado por `resumo/gerar_resumo_docx.py` | saída regenerável; o gabarito `modelo_mc2t_2026.docx` está versionado |

Os checkpoints são o ativo mais frágil do projeto — as Fases 0 e 2 quase se perderam quando
uma sessão do Colab caiu. Todo run espelha o `ckpt_best.pt` para fora do ambiente efêmero;
veja a Fase 2R no ROADMAP.

## Duas convenções que parecem inconsistência e não são

**As pastas em `runs/` mantêm os nomes em inglês** (`phase0_bossbase`, `phase2_residual`).
Esses nomes estão gravados dentro dos CSVs e dos logs já produzidos.
Renomeá-los invalidaria o registro histórico — é o mesmo motivo pelo qual a numeração
fracionária 2.5 foi preferida a renumerar as fases seguintes. Só os módulos `.py` foram
traduzidos.

**Os outputs de `notebooks/fase2_colab.ipynb` citam nomes antigos de arquivo.** É proposital:
aqueles outputs são o *único* registro do treino denso da Fase 2, e o
`ferramentas/logs_para_csv.py` extrai o log de treino de dentro deles. Editá-los seria
adulterar o registro.

## Notebooks e duplicação de código

Os notebooks do Colab são autossuficientes: contêm células `%%writefile` que gravam os
módulos no ambiente, sem precisar de upload. Isso duplica o código-fonte entre `.py` e
`.ipynb`, e as cópias já divergiram duas vezes. Nesta organização as células são
**cópia byte a byte de `modelo/` e `ferramentas/`**, que passam a ser a fonte única.

Ao alterar um módulo, regenere as células antes de rodar no Colab — ou troque os
`%%writefile` por um `git clone` deste repositório, que elimina a duplicação de vez.

## Referência

Este trabalho dá continuidade ao **STEGS2GAN** (Souza; Correia; Trevisani, 2026), do qual
incorpora os blocos residuais no gerador (Fase 2.5) e ao qual acrescenta o treinamento por
Wasserstein — ali listado como proposta futura — e a avaliação com esteganalisador
independente como etapa obrigatória.
