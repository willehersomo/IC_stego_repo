# Roadmap de fases — IC esteganografia com GANs

Princípio do projeto: construir em camadas. Cada fase adiciona uma peça sobre uma base
que já funciona, para que qualquer instabilidade de convergência tenha uma causa única e
identificável. Nunca tudo de uma vez.

Cada fase abaixo declara: o que muda, de que depende, e os critérios de saída  (as
condições que precisam valer antes de passar para a seguinte).

> **Revisão de 07/09/2026.** Quatro decisões de escopo entraram nesta versão:
> (1) o gerador ganha blocos residuais no modelo do STEGS2GAN — nova Fase 2.5;
> (2) o esteganalisador do laço da Fase 3 deixa de ser fixado na SRNet e passa a ser um
> detector **moderno**; (3) a Fase 4 passa a medir com **dois** detectores independentes,
> um moderno e a SRNet como linha comparável com a literatura; (4) a Fase 8 (robustez a
> compressão) deixa de ser opcional e vira entrega do trabalho.
> A numeração fracionária (2.5) é deliberada: renumerar 3→4, 4→5 … invalidaria todo log,
> CSV e nome de pasta já produzidos.

> **Revisão de 10/09/2026.** A **Fase 7** foi decomposta. Ela mudava dataset, cor, resolução e
> escala num único passo — quatro variáveis de uma vez, contra o princípio de camadas que rege
> o resto do roadmap. Agora são quatro passos (7a–7d), com DIV2K como porta de entrada e COCO
> por último. Ficou explícito também que o gate da fase é a **Fase 4 fechada**, não a
> numeração: sem `P_E` medido, treino em outro conjunto produz PSNR por dataset, que não
> sustenta afirmação sobre generalização. Até lá, todo treino novo é em BOSSBase.

> **Revisão de 10/09/2026 (2).** A **Fase 2.5 foi executada** e o gerador **residual venceu**:
> 38,27 dB contra 36,30 dB da Fase 2 densa, acurácia empatada, e L∞ quase pela metade. Registro
> completo na seção da fase. Duas coisas ficaram abertas e viraram trabalho: o **critério 3**
> (custo/época dos dois no mesmo hardware) não foi medido, e o residual tem **2,13× mais
> parâmetros**, o que confunde topologia com capacidade — daí a **linha de controle nova na
> Fase 5**. A partir daqui, "o gerador" do projeto é o residual.

---

## Situação atual

| Fase | O que é | Status |
|---|---|---|
| 0 | Codificador + Decodificador, sem adversário | fechada |
| 1 | Recuperação ~100% (mesmo código da Fase 0) | fechada |
| 2 | WGAN-GP — 1º treino adversarial (crítico de realismo) | fechada |
| 2R | Reconstituição dos checkpoints | dispensada — estão no Drive |
| **2.5** | **Blocos residuais no gerador (STEGS2GAN)** | **executada 10/09 — residual vence**; critério 3 em aberto |
| 3 | Esteganalisador moderno no laço — 2º adversário | **1º run (SRNet) reprovado** (20/08) — ver abaixo |
| 4 | Esteganalisadores independentes (P_E) | nova |
| 5 | Baselines, ablações e curva segurança × capacidade | nova |
| 6 | Carga adaptativa por atenção | nova |
| 7 | Generalização e escala (DIV2K / COCO / resolução maior) | nova |
| 8 | Robustez a compressão (JPEG) — muda o modelo de ameaça | **obrigatória** (era opcional) |

Resultados já medidos (BOSSBase, grayscale 128×128, 1024 bits/img = 0,0625 bpp):

Gerador **denso**:

- **Fase 0/1** (melhor ép. 109/120): acurácia 0,9999 · PSNR 35,16 dB · SSIM 0,963
- **Fase 2** (melhor ép. 55/60): acurácia 0,9997 · PSNR 36,30 dB · SSIM 0,974 · W subiu de
  0,275 (ép. 1) até 0,85–0,92 no miolo do treino e assentou em 0,813–0,828 nas 10 últimas épocas
  (média das 60 épocas 0,859; média das 10 últimas 0,822; máximo 0,922 na ép. 15)

Gerador **residual** (Fase 2.5, run de 10/09/2026):

- **Fase 0** (melhor ép. 113/120): acurácia 1,0000 · PSNR 35,94 dB · SSIM 0,9467 · L∞ 0,2336
- **Fase 2** (melhor ép. 59/60): acurácia 0,9998 · PSNR **38,27 dB** · SSIM 0,9803 · L∞ 0,2028 ·
  W de +0,284 (ép. 1), máx. +1,042 (ép. 5), média +0,859 nas 10 últimas
- **Fase 3** (run de 20/08, SRNet, `lambda_srnet=0,01`, 60 ép.): **reprovado no critério 3.**
  AUC do laço cravada em 1,0000 em 55/60 épocas (mínima 0,9978, na época 1).

---

## Fase 2R — Reconstituição dos checkpoints — DISPENSADA

Os pesos das Fases 0 e 2 não estão em disco local: sobrevivem só `runs/phase0/ckpt_last.pt`
(época 2) e `runs/phase0_bossbase/ckpt_last.pt` (época 8), ambos de 07/07 — runs iniciais, não
os finais. Do Colab voltaram os CSVs e os PNGs, não os `.pt`.

**Resolvido sem re-treinar:** os checkpoints bons estão no **Google Drive**. Basta apontar
`carregar_fase2` do `modelo/fase3_esteganalisador.py` para o `ckpt_best.pt` de lá — a célula 5 do
`fase3_colab.ipynb` varre o Drive e lista os candidatos com época, chaves e PSNR.

> **Pendência aberta.** O `CONFIG` do `modelo/fase3_esteganalisador.py` ainda tem
> `carregar_fase2 = "runs/phase2_bossbase/ckpt_best.pt"`, caminho que **não existe** em disco
> local — só `metricas_val.csv` e a curva estão lá. Fora do Colab, qualquer run da Fase 3 parte
> de peso nenhum sem avisar. Conferir antes de rodar.

> **Registro da Fase 2 recuperado em 07/09/2026.** O treino da Fase 2 nunca gravou `.log`
> nem `metricas_treino.csv` — o único registro era a saída das células do `fase2_colab.ipynb`,
> que não estava no projeto. O notebook foi trazido para cá e o `logs_para_csv.py` extrai dele
> `runs/phase2_bossbase/train_colab.log` e os dois CSVs, agora **com a coluna `w_dist`**, que
> é o que sustenta (e corrigiu) a afirmação sobre o W acima. Notebook sem output salvo é
> registro perdido: o `fase0_colab.ipynb` e o `fase3_colab.ipynb` que estão no projeto têm
> zero outputs.

> Lição que fica, e que já está implementada: **todo run espelha o checkpoint**. O
> `modelo/fase3_esteganalisador.py` tem `espelho_ckpt`, que copia o `ckpt_best.pt` para fora do ambiente
> efêmero a cada melhoria — não só no fim. Curva sem peso é resultado que não se pode continuar.

---

## Fase 2.5 — Blocos residuais no gerador

**O que muda.** A topologia interna do codificador. Hoje ele é **denso**, no estilo DenseNet:
cada bloco recebe a concatenação das features de todos os anteriores mais a mensagem
reinjetada.

```python
b = self.bloco2(torch.cat([a, mensagem], dim=1))
c = self.bloco3(torch.cat([a, b, mensagem], dim=1))
```

O STEGS2GAN usa três **blocos residuais** de verdade (`x + conv(x)`, com dois `Conv3x3+BN` no
caminho e `LeakyReLU` no meio) no `BasicEncoder`. Esta fase troca o interior do gerador por
essa topologia. O **skip global permanece**: o codificador continua aprendendo um resíduo
somado à portadora (`estego = portadora + residuo`) — isso nunca foi o "bloco residual" a que o
artigo se refere, e a confusão entre as duas coisas é o que produziu a ablação fantasma
corrigida abaixo.

**Correção de um erro do roadmap anterior.** A Fase 5 listava a ablação "com / sem blocos
residuais". Ela era impossível: não há blocos residuais no código para remover. A ablação
existia por leitura do artigo, não do código. Com esta fase ela passa a ser uma comparação
real — **denso × residual** — entre dois modelos que existem.

**Por que é fase e não linha de ablação.** Mudar a arquitetura do gerador invalida duas coisas
de uma vez: o `state_dict` da Fase 2 (warm start incompatível) e **todo número já medido**, que
foi medido com o gerador denso. Pelo princípio de camadas, isso é uma peça — e uma peça que
mexe na base, não no topo.

**Depende de.** Fase 2 fechada. **Não depende da Fase 3** — pode ser feita antes ou depois dela.

**Até quando dá para adiar.** Até imediatamente **antes da Fase 4**, e nem um passo além. A
razão é o que a Fase 4 é: o instrumento de medida. `P_E`, a curva segurança × capacidade da
Fase 5 e tudo que vier depois são medidos *sobre um gerador específico*. Trocar o gerador
depois disso obriga a refazer cada linha já medida, sob pena de violar a regra da própria
Fase 5 — *comparação com protocolo diferente entre linhas não é comparação*. Antes da Fase 4 o
custo é um treino; depois, é o conjunto inteiro de resultados.

Adiar até depois da Fase 3 tem inclusive um argumento a favor: o veredito H1/H2 (abaixo) diz
se o jogo adversarial é jogável neste payload, e essa resposta muda o que se pede do gerador.

**Cuidados.**
- **Warm start não existe aqui.** O `ckpt_best.pt` da Fase 2 não carrega num gerador de
  arquitetura diferente. Orçar o custo real: é refazer o caminho 0 → 2 com o gerador novo, não
  um fine-tuning.
- **Uma variável por vez.** Mesmo dataset, mesmo payload, mesmas épocas, mesmo `lambda_adv`,
  mesma semente. Se mudar largura e topologia juntas, o resultado não atribui a nenhuma das duas.
- Registrar **nº de parâmetros e s/época** dos dois geradores. O roadmap já registra que "o
  gargalo do projeto é o gerador denso, que roda a 128×128 sem reduzir resolução" — a troca
  pode pagar em tempo, e esse é um resultado a reportar mesmo que a qualidade empate.

**Critérios de saída.**
1. Acurácia de bits ≥ 0,99.
2. PSNR ≥ 36,30 dB — não regredir o que a Fase 2 entregou. Se cair, documentar quanto e decidir
   explicitamente, não por omissão.
3. Custo por época dos dois geradores medido e comparado, no mesmo hardware.
4. **Os dois geradores preservados em disco**, com checkpoint espelhado. A Fase 5 vai comparar
   os dois; um deles perdido significa a comparação perdida.

**Regra de decisão de qual segue.** Maior PSNR a igual acurácia. Empate técnico (< 0,2 dB) →
o mais barato por época. Registrar a decisão e o motivo — não deixar implícito no código.

---

### Registro do run (10/09/2026) — residual aprovado, critério 3 em aberto

`largura=64`, `tamanho_msg=32`, `tamanho_lote=32`, `lambda_imagem=2,0`, `lambda_residuo=0,05`,
`lambda_adv=0,01`, `n_critic=5`, `BLOCOS_RESIDUAIS=3`. Fase 0 residual 120 ép. → warm start →
Fase 2 residual 60 ép., ambas completas.

| | Fase 0 denso | Fase 0 residual | Fase 2 denso | **Fase 2 residual** |
|---|---|---|---|---|
| melhor época | 109/120 | 113/120 | 37/60 ¹ | **59/60** |
| acurácia de bits | 0,9999 | 1,0000 | 0,9997 | **0,9998** |
| PSNR | 35,16 dB | 35,94 dB | 36,30 dB | **38,27 dB** |
| SSIM | 0,9625 | 0,9467 | 0,9731 | **0,9803** |
| L∞ | 0,3622 | 0,2336 | 0,3878 | **0,2028** |
| épocas ≥ 35 dB | 1/120 | 35/120 | 56/60 | **60/60** |
| épocas ≥ 36,30 dB | 0/120 | 0/120 | 3/60 | **60/60** |

¹ empate triplo em 36,30 dB nas épocas 37, 54 e 55 — o texto acima registra a 55, a regra do
código seleciona a primeira (`score > melhor`). Mesmo PSNR nas três.

**Decisão: segue o gerador residual.** +1,97 dB de PSNR sobre a Fase 2 densa com a acurácia
empatada; não é empate técnico (< 0,2 dB), então o desempate por custo não é acionado.

**O resultado não é um pico.** As **60** épocas do residual ficam acima de 36,30 dB, que era o
*melhor ponto isolado* do gerador denso. O patamar inteiro subiu, não a sorte de uma época.

**O número a levar para as Fases 3 e 4: L∞ caiu de 0,3878 para 0,2028**, quase metade. É a
amplitude máxima da perturbação por pixel — exatamente o que um esteganalisador procura. O
gerador residual esconde os mesmos 1.024 bits com metade da amplitude de resíduo, o que é um
bom presságio para o `P_E` e **pode mudar o veredito H1/H2 da Fase 3**, levantado com o denso.

**Estabilidade adversarial sem surpresa.** W partiu de +0,284 (denso: +0,275), picou em +1,042
na ép. 5 (denso: +0,922 na ép. 15) e assentou em +0,859 nas 10 últimas (denso: +0,822). Mesmo
regime, sem divergência. O termo `adv` sobe a +12,04 (denso: +8,11) — o crítico pontua a estego
cada vez mais para baixo nos dois braços, mas o PSNR seguiu subindo até a ép. 59, então não é
divergência. Vale reolhar isso na Fase 3, onde há um segundo adversário.

**Confundidor a declarar: o residual tem 2,13× mais parâmetros.**

| | codificador | decodificador | gerador |
|---|---|---|---|
| denso | 114.634 | 113.473 | 228.107 |
| residual | 262.345 ² | 224.584 ² | **486.929** ² |

² Números impressos pela seção 5 do `fase25_colab.ipynb` na sessão de 10/09, que é o único
registro deles — aquele notebook foi salvo sem outputs e o `treino.log` não grava contagem de
parâmetros. O `modelo/fase25_residual.py` de hoje dá **261.313 · 223.681 · 484.994**: o código
mudou depois do run. A razão não se mexe (2,126 contra 2,135), então nada do que se conclui
abaixo depende de qual das duas contagens se usa.

A igual `largura=64`, a topologia residual naturalmente gera mais parâmetros — é assim que a
arquitetura do STEGS2GAN é definida, e comparar a igual `largura` é protocolo legítimo. Mas a
comparação **confunde topologia com capacidade**: o ganho de 1,97 dB pode ser "blocos residuais
ajudam", "o dobro de parâmetros ajuda", ou os dois. Ver a linha de controle acrescentada à
Fase 5.

Registrar também que a troca pegou **codificador e decodificador** (`modelo/fase25_residual.py:14-63`),
não só o codificador. É coerente com "o gerador" (= codificador + decodificador), mas a abertura
desta fase enfatiza o codificador; a redação e o que foi feito precisam dizer a mesma coisa.

**Critérios de saída — situação.**

| | Critério | Situação |
|---|---|---|
| 1 | acurácia ≥ 0,99 | ✅ 0,9998 |
| 2 | PSNR ≥ 36,30 dB | ✅ 38,27 dB (+1,97) |
| 3 | custo/época dos dois, mesmo hardware | ❌ **em aberto** |
| 4 | os dois geradores preservados | ⚠️ residual sim; conferir o denso no Drive |

**O critério 3 não fecha.** O residual está medido — **564,6 s/época, 9,4 h nas 60 épocas** —,
mas o run denso da Fase 2 é anterior à linha `[tempo]`, que é uma feature do
`modelo/fase25_residual.py`. Não há número denso comparável, e mesmo que houvesse seria outra sessão
de Colab, possivelmente outra GPU. Isso importa porque esta fase especulava que a troca "pode
pagar em tempo": pela ordem de grandeza (o run da Fase 3, denso + SRNet, rodou a ~305 s/época),
o residual parece **mais caro, não mais barato**. Resolver é barato: rodar o braço denso por 2–3
épocas na mesma sessão, só para colher s/época.

---

## Fase 3 — Esteganalisador moderno no laço adversarial

**O que muda.** Entra o segundo adversário. O crítico WGAN-GP cuida do *realismo*; o
esteganalisador cuida da *indetectabilidade estatística* — é um detector de verdade, projetado
para achar micro-ruído residual. O gerador passa a ter três termos de perda: mensagem, imagem e
agora dois adversários com pesos separados.

**Mudança em relação à versão anterior deste roadmap.** O detector do laço **não é mais fixado
na SRNet** (2019). Passa a ser um esteganalisador moderno, a escolher.

**Escolha do detector — com revisão de literatura, não de memória.** Candidatos a levantar:
ZhuNet, GBRAS-Net, SiaStegNet, LWENet e as linhas com transformer (CTNet e afins). Conferir
datas, resultados reportados e disponibilidade de código antes de fixar. Critérios de seleção:

1. publicado com código, e reprodutível;
2. treinável **do zero** dentro do orçamento de GPU do projeto (o warm start é do gerador; um
   detector pré-treinado contra outro gerador entra viciado);
3. **diferenciável ponta a ponta em relação à entrada** — o gradiente precisa chegar no gerador.
   Camadas de pré-processamento com filtros SRM fixos são aceitáveis (são convoluções com peso
   congelado); qualquer etapa não diferenciável, não;
4. custo por passo comparável ao da SRNet (~+22% no tempo por passo com `n_srnet=1`, medido).

**Pré-requisito que não mudou: rodar a sonda H1/H2 antes de escolher.** A sonda da seção 6.5 do
`fase3_colab.ipynb` usa a SRNet, e continua valendo, porque a pergunta dela é sobre **o jogo**,
não sobre o detector: se a AUC não sai de 1,0000 nem com `lambda_srnet = 2,0` contra a SRNet,
então trocar por um detector *mais forte* **piora** a situação, não melhora. Rodar a sonda antes
é o que separa trocar o detector por um motivo de trocar por otimismo. Custo: ~1 h 45.

**Depende de.** Gerador definido (Fase 2.5, se ela vier antes) + veredito da sonda H1/H2.

**Cuidados.**
- `lambda_srnet` (renomear para `lambda_stega` se o detector mudar) é o knob principal, e o
  mesmo trade-off da Fase 2 vale: alto demais derruba a acurácia de bits, baixo demais não muda
  nada.
- Warm start do gerador, mas detector **do zero**.
- Avaliar sempre com portadora e estego **no mesmo lote** — `avaliar_srnet` já trata a BatchNorm
  de propósito, usa Mann-Whitney e imagens pareadas. Qualquer detector novo herda essa exigência.

**Critérios de saída.**
1. Acurácia de bits ≥ 0,99 (não regredir o que a Fase 1 conquistou).
2. PSNR não cair mais que ~1 dB em relação ao gerador de partida (36,30 dB da Fase 2, ou o
   número que a Fase 2.5 fechar).
3. O detector do laço estacionar perto do acaso (AUC ≈ 0,5).

> **Armadilha, e a razão de a Fase 4 existir.** O critério 3 mede o detector que *participou
> do treino*. Ele pode estar em 0,5 porque o gerador ficou bom, ou porque aquele detector
> específico não convergiu. Um detector do laço em AUC 0,5 **não é evidência de segurança** —
> é evidência sobre aquele adversário. A prova real é a Fase 4. Trocar a SRNet por um detector
> moderno **não afrouxa** essa armadilha; ela vale para qualquer detector do laço.

---

### Registro do 1º run (20/08/2026, com SRNet) — reprovado

`lambda_srnet=0,01`, `n_srnet=1`, SRNet cheia (2,42M par.), warm start da Fase 2 (ép. 55),
60 épocas, ~305 s/época, gerador denso.

| | ép. 1 | méd. ép. 51–60 | |
|---|---|---|---|
| AUC do laço | 0,9978 | **1,0000** | piorou |
| PSNR | 36,25 dB | 34,47 dB | −1,8 dB |
| SSIM | 0,9739 | 0,9473 | −0,027 |
| `srnet_bce` | 0,2364 | ~0,0012 | detector dominou |
| `enganar` | 1,95 | ~9,3 | gerador perdendo mais |
| `W` | +0,801 | +0,984 (máx 1,056, ép. 50) | a Fase 2 terminou em ~0,815 |

Só 24 das 60 épocas passam no critério 2 (PSNR ≥ 35,3). O checkpoint selecionado foi o da
**época 1** — ou seja, a Fase 2 quase intacta. **Não há modelo da Fase 3.**

O treino trocou 1,8 dB de PSNR e 0,027 de SSIM por ganho nulo de indetectabilidade.

**A AUC de 1,0 não é artefato.** `avaliar_srnet` põe portadora e estego no mesmo lote (a
BatchNorm está tratada de propósito), usa Mann-Whitney e imagens pareadas. Foi conferido.

> **A conta que estava errada e induziu o `lambda_srnet=0,01`.** A rodada de sanidade
> concluiu que "o termo da SRNet já é 64% da perda do gerador, faltam passos e não peso".
> O valor de fato é grande, mas isso não diz nada sobre a força do empurrão:
> `BCE(z, 0.5) = 0,5·[softplus(−z) + softplus(z)] ≈ 0,5·z` cresce **linearmente** no logit,
> enquanto seu gradiente `σ(z) − 0,5` **satura em 0,5**. Com `enganar ≈ 9,3` temos z ≈ 18,6 e
> gradiente exatamente 0,5 — peso efetivo `0,01 × 0,5 = 0,005`. **Ler magnitude de perda como
> se fosse magnitude de gradiente é o erro a não repetir.**

**Hipóteses abertas**, testadas pela sonda da seção 6.5 do `fase3_colab.ipynb`
(dois braços de 10 épocas, `lambda_srnet ∈ {0,5 · 2,0}`, ~1 h 45) — **ainda não executada**:

- **H1 — peso baixo demais.** Peso efetivo de 0,005 nunca remodelou o resíduo.
  → continuar subindo `lambda`, rodar as 60 épocas com o valor que mostrou movimento, e a
  troca por um detector moderno passa a fazer sentido: o jogo é jogável.
- **H2 — jogo invencível aqui.** O detector re-vence a cada passo e nenhum peso resolve; o que
  é detectável é o resíduo uniforme a 0,0625 bpp. O sinal a favor: `enganar` **sobe** de 1,95
  para 9,3 — o detector fica mais confiante, o jogo diverge.
  → a saída não é tuning **nem detector melhor**. É sonda de payload (`tamanho_msg=16`,
  0,0156 bpp) e, depois, decidir *onde* os bits vão — que é a Fase 6.

**Dois bugs corrigidos em 25/08:**
- `_espelhar` levantava `SameFileError` a cada época quando `espelho_ckpt == pasta_saida`
  (caso comum rodando direto no Drive). Poluiu o log com ~240 linhas e, pior, deixou **sem
  exercício** justamente o mecanismo criado para não perder pesos de novo. Agora retorna
  silenciosamente quando os caminhos coincidem.
- O critério 2 usava 35,0 na marca por época e 35,3 no veredito final. **Treze** épocas caem
  na faixa e apareciam aprovadas no log (`criterios[12.]`) para reprovar no fim: 7 (35,06),
  12 (35,23), 13 (35,09), 15 (35,17), 30 (35,22), 32 (35,03), 34 (35,03), 35 (35,02),
  36 (35,26), 39 (35,18), 42 (35,07), 44 (35,19) e 52 (35,02). Agora há fonte única em
  `cfg["alvo_psnr"] = 35.3`.

---

## Fase 4 — Esteganalisadores independentes (P_E)

**O que muda.** Nenhum treino de gerador. Esta fase constrói o *instrumento de medida* que
o projeto ainda não tem — e sem o qual nenhuma das fases seguintes pode ser avaliada.

**Protocolo.**
1. Congelar o gerador da Fase 3.
2. Gerar um corpus pareado (portadora, estego) sobre um conjunto de imagens **que não foi
   usado no treino do gerador**. Além do split retido do BOSSBase, gerar também um corpus com
   **DIV2K e COCO** passando pelo gerador congelado — só inferência, sem re-treino. É a sonda
   de generalização barata da Fase 7; ver a seção daquela fase.
3. Treinar do zero detectores que nunca participaram do laço. **Três variantes**, porque
   respondem perguntas diferentes:
   - **mesma família, outra semente** (o detector do laço reinicializado) — mede se o gerador
     aprendeu a enganar *aquela instância* ou a *classe de detectores*;
   - **família diferente e moderna** — o estado da arte que ficou de fora do laço. É o número
     mais forte que a fase produz: indetectabilidade que sobrevive ao melhor detector conhecido;
   - **SRNet (2019)** — não porque seja moderna, mas porque é a **linha comparável**: quase todo
     trabalho de esteganografia por aprendizado reporta contra ela, inclusive o artigo do SBSeg.
     Sem essa linha o resultado não conversa com a literatura nem com o trabalho anterior.
   - *(opcional)* **SRM + ensemble clássico** — detector não-neural, viés indutivo totalmente
     distinto. Barato e difícil de contestar.
4. Medir no conjunto de teste.

**Métrica.** `P_E = min_limiar (P_FA + P_MD)/2` — o erro do detector no seu melhor limiar.
`P_E ≈ 0,5` é indetectável; `P_E → 0` é detectado. Reportar junto a AUC, **por detector** —
uma tabela, nunca um número só.

**Disciplina que não pode ser relaxada.**
- Separação por imagem: a mesma portadora nunca pode aparecer no treino e no teste do
  detector, nem em lados opostos do par.
- **AUC abaixo de 0,5 não é sucesso.** A métrica é simétrica: significa que o detector
  aprendeu a separação com os rótulos invertidos, e basta inverter a regra de decisão.
  Ao reportar, sempre considerar `max(AUC, 1-AUC)`.
- Mesmo split, mesmo nº de épocas e mesmo orçamento de treino para todos os detectores. Um
  detector mal treinado produz `P_E` alto que parece segurança e é só sub-treino.

**Critérios de saída.** Ter os números medidos, com o protocolo acima documentado. Esta fase
não tem alvo a bater — ela produz o valor honesto, seja ele qual for. É a métrica que valida
(ou não) a tese do trabalho, e a limitação nº 1 registrada no artigo do SBSeg.

---

## Fase 5 — Baselines, ablações e curva segurança × capacidade

**O que muda.** Nada na arquitetura. Esta fase responde "o que de fato contribuiu?" usando
o instrumento da Fase 4 em todas as comparações.

**Baselines** (o que o trabalho ainda não tem, e que o artigo lista como limitação):
- LSB clássico, no mesmo payload;
- Fase 0 pura, sem adversário nenhum;
- Fase 2, só WGAN-GP;
- Fase 3, WGAN-GP + esteganalisador no laço.

**Ablações e comparações** (isolar cada peça, uma de cada vez):
- **gerador denso × gerador residual** — a comparação real que a Fase 2.5 torna possível.
  Os dois checkpoints têm que ter sobrevivido; ver critério de saída 4 daquela fase;
- **controle de capacidade: denso com `largura` elevada até ~485k parâmetros no gerador** —
  linha nova, exigida pelo resultado da Fase 2.5. A igual `largura=64` o residual tem 2,13×
  mais parâmetros que o denso (484.994 contra 228.107, pelo código de hoje), então o ganho
  de 1,97 dB medido ali confunde *topologia* com *capacidade*. Sem esta linha, a ablação
  denso × residual não sustenta afirmação sobre topologia. Se o residual ainda vencer a
  igual nº de parâmetros, a afirmação é sobre topologia; se empatar, era capacidade — e
  isso também é resultado. **Rodar antes da Fase 4**, que é onde o gerador escolhido
  congela;
- com / sem crítico WGAN-GP;
- com / sem esteganalisador no laço.

> A ablação "com / sem blocos residuais" listada na versão anterior deste roadmap **foi
> removida**: não havia blocos residuais no código para remover. Virou a comparação
> denso × residual acima.

**Varredura de payload** — retomar a que foi interrompida pela queda do Colab.
`tamanho_msg ∈ {16, 32, 64, 128}`, ou seja bpp de 0,0156 a 1,0. Produz a **curva
segurança × capacidade**: `P_E` no eixo y, `bpp` no eixo x, com PSNR anotado. É o resultado
mais forte que um trabalho de esteganografia pode apresentar, porque mostra o trade-off
inteiro em vez de um ponto isolado.

**Critérios de saída.** Tabela completa e curva, todas as linhas medidas com o mesmo
protocolo da Fase 4 — mesma família de detector, mesmo split, mesmo número de épocas.
Comparação com protocolo diferente entre linhas não é comparação.

---

## Fase 6 — Carga adaptativa por atenção

**O que muda.** Hoje a mensagem é espalhada uniformemente e o payload é fixo. Regiões lisas
(céu, parede) e regiões texturizadas (folhagem, cabelo) recebem a mesma carga, embora só as
segundas escondam bem. Esta fase deriva um **mapa de custo/atenção** da complexidade
textural local e usa esse mapa para decidir *onde* colocar os bits.

**Motivação direta.** É a limitação "carga estática não aproveita as regiões texturizadas"
registrada no artigo do SBSeg, e o trabalho futuro de "mecanismos de atenção visual". Se a
sonda da Fase 3 fechar em **H2**, esta fase deixa de ser incremento e passa a ser a resposta
principal do trabalho — o que é detectável é o resíduo uniforme, e é exatamente isso que ela ataca.

**Depende de.** Fase 4 — sem P_E medido, não há como saber se a atenção ajudou.

**Critérios de saída.** Uma das duas, medida contra a Fase 3 na curva da Fase 5:
- mesmo bpp com `P_E` mais alto (mais indetectável); **ou**
- bpp maior com o mesmo `P_E`.

Se nenhuma valer, o resultado negativo também é resultado — e vale registrar no relatório.

---

## Fase 7 — Generalização e escala

**O que muda.** Sair do BOSSBase. Ao final da fase o trabalho terá treinado em DIV2K e COCO
(RGB, `canais=3`) e a 256×256. Os três conjuntos já estão em disco, e as diferenças entre eles
não são só de conteúdo:

| Conjunto | Tamanho | Formato | Natureza |
|---|---|---|---|
| `datasets/BOSSbase_1.01` | 10.000 | `.pgm` | grayscale, câmera nativa, **nunca comprimido** |
| `datasets/DIV2K_train_HR` | 800 | `.png` | RGB, 2K, **sem perdas** |
| `datasets/val2017` (COCO) | 5.000 | `.jpg` | RGB, **já comprimido em JPEG** |

**Por que importa.** É o trabalho futuro de "expansão do conjunto de dados" do artigo, e
tem uma hipótese concreta a testar: no STEGS2GAN, o COCO induziu **colapso de modo** enquanto
o DIV2K convergiu. A pergunta desta fase é se o WGAN-GP — que o STEGS2GAN não tinha — evita
esse colapso. Se evitar, é uma contribuição publicável por si só.

**Depende de: Fase 4 fechada.** Este é o gate real da fase, não a posição na numeração. O
critério de saída 2 pede `P_E` por conjunto: sem o instrumento da Fase 4, um treino em DIV2K
produz PSNR e SSIM por dataset, e isso não sustenta afirmação nenhuma sobre generalização —
um gerador pode entregar 37 dB em DIV2K e ser trivialmente detectável. **Até a Fase 4 fechar,
todo treino novo é em BOSSBase**, que é onde os números conversam com a literatura
(S-UNIWARD, WOW, HILL e SRNet são reportados nele) e entre si.

**Sonda barata, a fazer dentro da Fase 4.** O passo 2 do protocolo da Fase 4 pede um corpus
pareado sobre imagens não usadas no treino do gerador. Aproveitar para gerar, além do split
retido do BOSSBase, um corpus com DIV2K e COCO passando pelo **gerador congelado** — só
inferência, sem re-treino — e medir acurácia de bits, PSNR e `P_E` fora do domínio. Custo:
algumas horas de GPU e um treino de detector a mais. Não responde a hipótese do colapso de
modo, que é sobre *treinar* em COCO e não sobre inferir; mas se a acurácia de bits já cair na
inferência, a dificuldade da Fase 7 aparece antes de qualquer treino ser comprometido.

**Ordem — uma variável por vez.** A versão anterior desta fase mudava dataset, cor, resolução
e escala de uma só vez, o que contradiz o princípio do projeto: se o resultado piorasse, não
haveria como atribuir a piora a nenhuma das quatro. A fase passa a ter quatro passos:

| Passo | O que muda | Variável isolada |
|---|---|---|
| 7a | DIV2K **convertido para grayscale**, 128×128 | estatística de imagem |
| 7b | DIV2K RGB (`canais=3`), 128×128 | cor |
| 7c | DIV2K RGB, 256×256 | resolução |
| 7d | COCO | JPEG na origem + hipótese do colapso de modo |

Se o orçamento de GPU não der para os quatro, fundir 7a e 7b (DIV2K RGB direto) e manter 7c e
7d separados — resolução e COCO são os dois que mais provavelmente quebram alguma coisa, e
portanto os que mais se precisa conseguir atribuir.

**Cuidados.**
- **DIV2K tem 800 imagens contra 10.000 do BOSSBase.** Recortes aleatórios de 128×128 sobre
  imagens 2K dão amostras de sobra, mas o número de *cenas distintas* é 12× menor — e isso é
  um confundidor direto numa afirmação de generalização: um resultado pior em DIV2K pode ser
  diversidade de cena, não domínio. Subamostrar o BOSSBase para 800 imagens numa das linhas,
  ou reportar a diferença explicitamente.
- **COCO já é JPEG na origem; a Fase 8 é JPEG no canal — são coisas diferentes.** A Fase 8
  simula a compressão que a estego sofre *depois* de transmitida; o COCO traz compressão na
  *portadora*, antes de qualquer coisa acontecer. Juntas, tornam o resultado da Fase 8
  ilegível: não se separa "o resíduo sobrevive ao JPEG do canal" de "o resíduo interage com o
  JPEG que já estava na origem". Por isso 7d é o último passo, de preferência **depois** de a
  Fase 8 ter medido o efeito do canal sobre portadoras limpas.
- **COCO é o caso difícil, não a porta de entrada.** O registro do STEGS2GAN é de colapso em
  COCO e convergência em DIV2K. Entrar por DIV2K é entrar pelo caso que se sabe que converge.
- Em `.jpg` a portadora carrega uma assinatura de quantização conhecida, e o resíduo do
  gerador perturba justamente essa estrutura — é dos sinais mais fáceis que um esteganalisador
  aprende. Esperar `P_E` pior em COCO por esse motivo, e não confundir com colapso de modo.

**Critérios de saída.**
1. Treinar nos três conjuntos sem divergência, ou documentar a divergência com os três
   indicadores: perda de validação, comportamento de W (ou da perda adversarial) e queda
   de qualidade visual.
2. Reportar `P_E` por conjunto — generalização se mede na métrica de segurança, não no PSNR.
3. Cada passo 7a–7d medido com o mesmo protocolo da Fase 4 e reportado separadamente. Passo
   fundido é passo cuja variável não se atribui: se 7a e 7b forem fundidos, registrar a fusão
   junto do resultado.

---

## Fase 8 — Robustez a compressão em canal real (JPEG)

**Deixou de ser opcional em 07/09/2026.** Passa a ser entrega do trabalho. A posição na ordem
não muda — ela continua depois da Fase 5, pelos motivos no fim desta seção.

**O que muda.** Uma camada de distorção entra entre o codificador e o decodificador durante o
treino: a estego é comprimida (JPEG) antes de o decodificador tentar ler os bits. O gerador é
forçado a esconder a mensagem onde ela **sobrevive à quantização**.

**Por que importa.** Hoje o decodificador lê a estego intacta. Qualquer canal real — WhatsApp,
Instagram, e-mail — recomprime. Uma esteganografia que não sobrevive a JPEG funciona no
laboratório e não funciona no mundo.

**Obstáculo técnico.** O JPEG não é diferenciável: a quantização é um arredondamento, cujo
gradiente é zero em quase todo ponto. Soluções, em ordem de esforço:
1. **Camadas de ruído aproximadas** (HiDDeN): `JPEG-mask` / `JPEG-drop`, que zeram os
   coeficientes DCT de alta frequência sem arredondar. Diferenciáveis por construção.
2. **JPEG diferenciável** com *straight-through estimator* no arredondamento — mais fiel.
3. Treinar com uma **mistura** de distorções (identidade, JPEG q ∈ {50, 75, 90}, blur, crop,
   dropout), sorteando uma por lote. Treinar só contra JPEG produz robustez só a JPEG.

**A tensão que define a fase.** Robustez e indetectabilidade **puxam em direções opostas**.
O JPEG descarta alta frequência; a indetectabilidade quer a perturbação exatamente ali, na
textura de alta frequência, onde o olho e a estatística não a encontram. Sobreviver à compressão
obriga a mover energia para frequências mais baixas e com amplitude maior — que é precisamente
o que um esteganalisador detecta. O trabalho passa a ter um trade-off de **três** eixos:
capacidade × robustez × indetectabilidade. Não dá para maximizar os três.

> Esta tensão ficou **mais apertada**, não menos, com a decisão de usar um detector moderno nas
> Fases 3 e 4: o que a robustez a JPEG obriga a fazer com o resíduo é exatamente o que um
> detector melhor acha com mais facilidade. Esperar `P_E` mais baixo aqui do que na Fase 5 é o
> comportamento previsto, não uma falha — o resultado da fase é *quanto* se paga.

**Mudança de modelo de ameaça — precisa ser declarada.** O modelo atual (e o do artigo do SBSeg)
diz que o adversário observa e **não altera** a imagem. Uma camada de compressão descreve um
canal que altera. Não é mais o mesmo cenário, e apresentar as duas coisas sob o mesmo modelo
de ameaça seria um erro. Ou se declara um canal ruidoso com adversário passivo, ou se assume
um *active warden*. No relatório final, as duas partes vão em seções separadas com o modelo de
ameaça declarado em cada uma.

**Depende de.** Fase 5. Sem a curva de segurança × capacidade já medida, não há como dizer
quanto de `P_E` cada nível de robustez custou — e esse custo é o resultado interessante.

**Critérios de saída.**
1. Acurácia de bits ≥ 0,95 após JPEG q=75 (limiar mais frouxo que o das outras fases: aqui
   perder bit é esperado, e é onde um ECC finalmente se justifica).
2. Reportar `P_E` **com e sem** a camada de compressão — a diferença é o preço da robustez.
3. Uma curva `P_E × qualidade JPEG`, não um ponto isolado.

---

## Mapa: o que cada fase fecha

| Pendência registrada | Fase |
|---|---|
| Blocos residuais no gerador (STEGS2GAN) | **2.5** |
| Esteganalisador do laço (moderno, no lugar da SRNet) | **3** |
| Indetectabilidade com esteganalisador independente (P_E) | 4 |
| Comparação com a literatura via SRNet como detector de referência | **4** |
| Ablações com/sem cada camada, e denso × residual | 5 |
| Baselines externos (LSB, SteganoGAN) | 5 |
| Varredura de payload (interrompida) | 5 |
| Atenção visual / carga adaptativa | 6 |
| Generalização DIV2K / COCO | 7 (7a–7b · 7d) |
| Expansão do conjunto de dados, resolução maior | 7 (7c) |
| Robustez a compressão / canal real (JPEG) | **8 (obrigatória)** |
| WGAN para estabilidade | **já fechada na Fase 2** |

A última linha merece destaque: o artigo do SBSeg lista WGAN como trabalho *futuro*, e aqui
ela já está fechada e funcionando desde a Fase 2. Os dois trabalhos são metades
complementares da mesma arquitetura — o SBSeg tem a SRNet no laço e não tem WGAN; a IC tem
WGAN-GP e ainda não tem esteganalisador no laço. A Fase 3 é o ponto onde as duas se encontram,
e a Fase 2.5 é onde a IC absorve a peça de arquitetura que faltava.

---

## Ordem e por quê

`2.5 → 3 → 4 → 5 → 6 → 7a–7c → 8 → 7d`

A ordem não é arbitrária.

- A **Fase 2.5 pode nadar** entre a 2 e a 4 — antes ou depois da 3, tanto faz —, mas **não
  atravessa a Fase 4**. Depois que o instrumento de medida existe e a curva começa a ser
  levantada, trocar o gerador obriga a refazer tudo que já foi medido.
- A **Fase 4 vem antes de tudo que é experimento** porque é o instrumento de medida: sem
  `P_E`, as Fases 5, 6 e 7 produziriam números que ninguém sabe interpretar.
- A **Fase 5 vem antes da 6** porque é ela que diz quais peças já contribuem — não faz sentido
  acrescentar atenção antes de saber o que a topologia do gerador e cada adversário estão
  realmente entregando.
- A **Fase 8 fica por último** porque muda o modelo de ameaça e porque o resultado que ela
  produz — quanto de `P_E` se paga por robustez — só existe se a curva da Fase 5 já estiver medida.
- **O passo 7d (COCO) é a única coisa que atravessa a Fase 8**, e de propósito. O COCO traz
  JPEG na *portadora*; a Fase 8 mede JPEG no *canal*. Rodando 7d antes, o resultado da Fase 8
  fica ilegível — não se separa uma compressão da outra. Rodando depois, a Fase 8 já
  estabeleceu a linha de base sobre portadoras limpas, e o COCO vira o caso combinado, que é
  o que ele de fato é. Se o orçamento não permitir esperar, rodar 7d antes e **reportar o
  entrelaçamento como limitação** — não omitir.

**Próxima ação concreta:** rodar a sonda H1/H2 (célula 6.5 do `fase3_colab.ipynb`, ~1 h 45),
agora **com o gerador residual**, e apontando `carregar_fase2` para o `ckpt_best.pt` da Fase 2.5
(ép. 59) — não para o caminho denso anotado na Fase 2R, que nem existe em disco local. A sonda é
barata, é pré-requisito da escolha do detector da Fase 3, e o veredito muda o peso da Fase 6 no
trabalho inteiro. **O resultado pode não ser o mesmo do run denso de 20/08:** o L∞ caiu de 0,3878
para 0,2028, e era exatamente a amplitude do resíduo que sustentava a AUC cravada em 1,0000.

Duas tarefas pequenas que podem ir junto, na mesma sessão de GPU: colher **s/época do braço
denso** (2–3 épocas bastam) para fechar o critério 3 da Fase 2.5, e confirmar que o
`ckpt_best.pt` **denso** ainda está no Drive — critério 4 da mesma fase, e sem ele a ablação da
Fase 5 morre.
