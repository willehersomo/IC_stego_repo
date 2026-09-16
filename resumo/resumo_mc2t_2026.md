# Resumo expandido — MC²T 2026

> **Gerado por `gerar_resumo_docx.py --md`.** Não edite este arquivo à mão: mude as
> constantes do script e rode de novo, senão ele volta a divergir do `.docx`.
> O documento pronto é `resumo_mc2t_2026.docx`; renderizado, fecha em 2 páginas.

---

## Título

**Esteganografia em imagens digitais com redes adversárias generativas: contribuição do crítico WGAN-GP para a estabilidade do treinamento**

## Autores

Breno W. B. Correia¹, Kleber M. Trevisani²

1. Discente do Curso de Bacharelado em Ciência da Computação – IFSP – Câmpus Presidente Epitácio;

2. Docente – IFSP – Câmpus Presidente Epitácio.

E-mails: breno.wille@aluno.ifsp.edu.br, kleber@ifsp.edu.br

## Resumo

*Resumo - Este trabalho investiga a ocultação de mensagens em imagens digitais por redes adversárias generativas, em fases incrementais que isolam cada componente. Um par codificador-decodificador foi treinado no BOSSBase a 0,0625 bits por pixel e depois acoplado a um crítico WGAN-GP. O crítico elevou o PSNR de 35,16 dB para 36,30 dB e estabilizou o treinamento: 56 de 60 épocas acima de 35 dB, contra 1 de 120 sem ele. A troca do gerador denso por blocos residuais elevou o PSNR a 38,27 dB, com as 60 épocas acima do melhor ponto isolado do gerador denso. As etapas seguintes inserem um esteganalisador no laço adversarial e medem a indetectabilidade com um detector independente.*

**Palavras-chave: Esteganografia; Redes adversárias generativas; Esteganálise.**

## Introdução

A esteganografia oculta a própria existência de uma comunicação dentro de uma mídia de cobertura, diferindo da criptografia, que torna a mensagem ininteligível mas assume sua presença; a esteganálise é a contraparte, decidir se uma mídia carrega conteúdo oculto (Fridrich, 2009). Com o aprendizado profundo, ambas passaram a ser formuladas como um jogo entre redes neurais, no qual um gerador aprende a esconder e um detector a encontrar — formulação consolidada pelo HiDDeN (Zhu et al., 2018) e pelo SteganoGAN (Zhang et al., 2019).

Nessa formulação competem capacidade, imperceptibilidade e indetectabilidade estatística, e otimizar as três de uma vez produz instabilidade de convergência e impede atribuir um resultado ruim a um componente específico. O problema investigado é quanto cada componente adversarial contribui para a indetectabilidade, e a que custo em qualidade. O objetivo é construir e avaliar um modelo de esteganografia por GAN em camadas, acrescentando um componente por vez com critérios declarados antes de cada etapa.

A justificativa é metodológica: é comum reportar a taxa de erro do detector que participou do próprio treino, número que mede aquele adversário e não a segurança do método. O trabalho dá continuidade ao STEGS2GAN (Souza; Correia; Trevisani, 2026), incorporando o treinamento por Wasserstein — ali listado como proposta futura — e tratando a avaliação independente como etapa obrigatória.

## Metodologia

Os experimentos usam o BOSSBase 1.01 (Bas; Filler; Pevný, 2011), padrão da área: 10.000 imagens em tons de cinza redimensionadas para 128×128 pixels, sendo 9.500 para treino e 500 para validação. A carga é fixa em 1.024 bits por imagem, ou 0,0625 bits por pixel: uma mensagem binária de 32×32 é replicada espacialmente até a resolução da imagem, e o decodificador agrega as cópias redundantes pela média.

O codificador é uma rede convolucional densa que recebe imagem e mensagem e produz um resíduo somado à imagem original; o decodificador recupera os bits a partir do resultado. A perda combina erro quadrático médio sobre a imagem e entropia cruzada binária sobre os bits. Sobre essa base, cada fase acrescenta um componente e fixa seus critérios de saída antes do treinamento. A Fase 2, já executada, insere um crítico WGAN-GP (Arjovsky; Chintala; Bottou, 2017; Gulrajani et al., 2017) com penalidade de gradiente e cinco passos de crítico por passo do gerador. A Fase 2.5, também executada, troca o codificador e o decodificador densos por blocos residuais no modelo do STEGS2GAN (Souza; Correia; Trevisani, 2026). A fase seguinte, ainda a executar, insere um esteganalisador SRNet (Boroumand; Chen; Fridrich, 2019) no laço adversarial, exigindo AUC próxima de 0,5 sem perder mais de 1 dB de PSNR. Cada fase parte dos pesos da anterior, exceto a troca de arquitetura, que não admite warm start. As métricas são acurácia de bits, PSNR e SSIM (Wang et al., 2004) para qualidade, e a área sob a curva ROC (AUC) do esteganalisador para detecção, calculada com imagens pareadas, em GPU no Google Colab. A Figura 1 resume a arquitetura resultante ao fim da Fase 2.5.

![Figura 1. Arquitetura do modelo até a Fase 2.5. Em laranja, o que a Fase 2 acrescentou; em azul, os blocos cujo miolo a Fase 2.5 trocou.](figuras/arquitetura_fase25.png)

**Figura 1. Arquitetura do modelo até a Fase 2.5. Em laranja, o que a Fase 2 acrescentou; em azul, os blocos cujo miolo a Fase 2.5 trocou.** Fonte: Elaborada pelo autor (2026).

## Resultados parciais e esperados

A Tabela 1 apresenta as três fases já executadas.

**Tabela 1. Resultados das fases já executadas (BOSSBase, 128×128, 0,0625 bpp; 120 épocas na Fase 0/1 e 60 nas Fases 2 e 2.5).**

| Fase | Acurácia | PSNR (dB) | SSIM | Épocas ≥ 35 dB |
|---|---|---|---|---|
| 0/1 — sem adversário | 0,9999 | 35,16 | 0,963 | 1 de 120 |
| 2 — WGAN-GP | 0,9997 | 36,30 | 0,974 | 56 de 60 |
| 2.5 — residual | 0,9998 | 38,27 | 0,980 | 60 de 60 |

Fonte: Elaborada pelo autor (2026).

A recuperação da mensagem foi praticamente perfeita em todas as fases, o que desloca a discussão para a qualidade da imagem. O ganho da Fase 2 não está no pico, e sim na estabilidade: sem o crítico, o PSNR alcançou 35 dB em uma única das 120 épocas, com média de 33,87 dB nas dez últimas; com o crítico, 56 das 60 superaram o limiar, com média de 36,15 dB.

A Fase 2.5 trocou a topologia densa do gerador por blocos residuais e produziu o maior ganho isolado até aqui: 38,27 dB contra 36,30 dB, sem custo de acurácia, com as 60 épocas acima do melhor ponto do denso e a perturbação máxima por pixel caindo de 0,388 para 0,203 — o que é favorável à indetectabilidade. A topologia residual tem, porém, 2,13 vezes mais parâmetros a igual largura de camada: o ganho ainda não separa topologia de capacidade, e a ablação a igual número de parâmetros está prevista. A Figura 2 põe as duas arquiteturas na mesma linha do tempo: o gerador residual está acima do denso desde a Fase 0 e mantém a distância depois que o crítico entra.

![Figura 2. PSNR e SSIM dos geradores denso e residual ao longo das Fases 0 e 2, em linha do tempo contínua.](figuras/curva_denso_residual.png)

**Figura 2. PSNR e SSIM dos geradores denso e residual ao longo das Fases 0 e 2, em linha do tempo contínua.** Fonte: Elaborada pelo autor (2026).

Como resultados esperados, pretende-se: (i) inserir um esteganalisador no laço adversarial e medir quanto de qualidade de imagem custa a indetectabilidade; (ii) treinar um esteganalisador independente sobre imagens não vistas pelo gerador e reportar a probabilidade de erro P_E, que mede segurança e não o desempenho de um adversário específico; (iii) levantar a curva segurança × capacidade de 0,0156 a 1,0 bpp; e (iv) avaliar a robustez a recompressão JPEG por camadas de ruído diferenciáveis (Zhu et al., 2018).

## Conclusões

O protocolo em camadas permite atribuir cada resultado ao componente que o causou. O crítico WGAN-GP mostrou-se benéfico e sobretudo estabilizador: elevou o PSNR de 35,16 dB para 36,30 dB sem custo de acurácia e, mais importante, transformou um limiar atingido uma única vez em 120 épocas em um patamar sustentado por 56 de 60. A troca por blocos residuais somou outros 1,97 dB, ainda sem separar topologia de capacidade. A etapa seguinte insere o esteganalisador no laço adversarial. A decisão metodológica que orienta o restante do trabalho é que a taxa de erro do detector que participou do treino não é evidência de segurança: por isso a avaliação com um esteganalisador independente entrou no cronograma como etapa obrigatória, e não como complemento.

## Agradecimentos

Os autores agradecem ao IFSP – Câmpus Presidente Epitácio pela infraestrutura e suporte fornecidos.

## Referências

ARJOVSKY, M.; CHINTALA, S.; BOTTOU, L. **Wasserstein generative adversarial networks**. In: International Conference on Machine Learning (ICML). PMLR, 2017. p. 214-223. Disponível em: https://proceedings.mlr.press/v70/arjovsky17a.html.

BAS, P.; FILLER, T.; PEVNÝ, T. **“Break our steganographic system”: the ins and outs of organizing BOSS**. In: International Workshop on Information Hiding. Praga: Springer, 2011. p. 59-70. Disponível em: [completar].

BOROUMAND, M.; CHEN, M.; FRIDRICH, J. **Deep residual network for steganalysis of digital images**. IEEE Transactions on Information Forensics and Security, v. 14, n. 5, p. 1181-1193, 2019. Disponível em: [completar].

FRIDRICH, J. **Steganography in Digital Media: Principles, Algorithms, and Applications**. Cambridge: Cambridge University Press, 2009. Disponível em: https://books.google.com/books/about/Steganography_in_Digital_Media.html?id=wcAZ-QEthqkC.

GULRAJANI, I.; AHMED, F.; ARJOVSKY, M.; DUMOULIN, V.; COURVILLE, A. **Improved training of Wasserstein GANs**. Advances in Neural Information Processing Systems (NIPS), v. 30, p. 5767-5777, 2017. Disponível em: https://arxiv.org/abs/1704.00028.

SOUZA, J. P. B. S.; CORREIA, B. W. B.; TREVISANI, K. M. **STEGS2GAN - um modelo de aprendizado profundo para esteganografia e esteganálise em imagens digitais utilizando redes adversárias generativas**. In: Workshop de Trabalhos de Iniciação Científica e de Graduação (WTICG). Porto Alegre: SBC, 2026. Disponível em: [completar].

WANG, Z.; BOVIK, A. C.; SHEIKH, H. R.; SIMONCELLI, E. P. **Image quality assessment: from error visibility to structural similarity**. IEEE Transactions on Image Processing, v. 13, n. 4, p. 600-612, 2004. Disponível em: https://doi.org/10.1109/TIP.2003.819861.

ZHANG, K. A.; CUESTA-INFANTE, A.; XU, L.; VEERAMACHANENI, K. **SteganoGAN: High Capacity Image Steganography with GANs**. arXiv preprint arXiv:1901.03892, 2019. Disponível em: https://arxiv.org/abs/1901.03892.

ZHU, J.; KAPLAN, R.; JOHNSON, J.; FEI-FEI, L. **HiDDeN: Hiding Data With Deep Networks**. In: Proceedings of the European Conference on Computer Vision (ECCV). Munique, Alemanha: Springer, 2018. p. 657-672. Disponível em: https://arxiv.org/abs/1807.09937.
