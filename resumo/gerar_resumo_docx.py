#!/usr/bin/env python3
"""Preenche o modelo de resumo expandido da MC²T com o conteúdo do projeto.

Trabalha sobre uma CÓPIA do modelo e preserva a formatação original: cada parágrafo
tem seu texto trocado mantendo o primeiro run (que carrega fonte, tamanho e itálico),
e os parágrafos novos são clonados de um protótipo já formatado.

O título do modelo fica dentro de uma caixa de texto, duplicada pelo mecanismo de
compatibilidade do Word (mc:AlternateContent) — por isso as duas cópias são trocadas.

Uso:  python3 gerar_resumo_docx.py [caminho_do_modelo.docx]
Saída: resumo_mc2t_2026.docx
"""
import copy
import os
import shutil
import sys

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Cm, Pt, Twips

# O modelo é de DUAS COLUNAS: cada coluna tem 5244 twips (~9,25 cm). Tabela mais larga
# que isso é cortada na exportação — foi o que aconteceu na primeira geração.
LARGURA_COLUNA = 5244

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
AQUI = os.path.dirname(os.path.abspath(__file__))          # resumo/
RAIZ = os.path.dirname(AQUI)                                # raiz do repositório
MODELO_PADRAO = os.path.join(AQUI, "modelo_mc2t_2026.docx")  # gabarito do evento
SAIDA = os.path.join(AQUI, "resumo_mc2t_2026.docx")

# --------------------------------------------------------------------------- conteúdo

TITULO = ("Esteganografia em imagens digitais com redes adversárias generativas: "
          "contribuição do crítico WGAN-GP para a estabilidade do treinamento")

AREA = "A – Ciências Exatas e da Terra"
AUTORES = "Breno W. B. Correia¹, Kleber M. Trevisani²"
AFILIACAO_1 = ("1. Discente do Curso de Bacharelado em Ciência da Computação – IFSP – "
               "Câmpus Presidente Epitácio;")
AFILIACAO_2 = "2. Docente – IFSP – Câmpus Presidente Epitácio."
EMAILS = "E-mails: breno.wille@aluno.ifsp.edu.br, kleber@ifsp.edu.br"

RESUMO = ("Resumo - Este trabalho investiga a ocultação de mensagens em imagens digitais por "
          "redes adversárias generativas, em fases incrementais que isolam cada componente. Um "
          "par codificador-decodificador foi treinado no BOSSBase a 0,0625 bits por pixel e "
          "depois acoplado a um crítico WGAN-GP. O crítico elevou o PSNR de 35,16 dB para "
          "36,30 dB e estabilizou o treinamento: 56 de 60 épocas acima de 35 dB, contra 1 de 120 "
          "sem ele. A troca do gerador denso por blocos residuais elevou o PSNR a 38,27 dB, com as "
          "60 épocas acima do melhor ponto isolado do gerador denso. As etapas seguintes inserem "
          "um esteganalisador no laço adversarial e medem a indetectabilidade com um detector "
          "independente.")

PALAVRAS_CHAVE = "Palavras-chave: Esteganografia; Redes adversárias generativas; Esteganálise."

INTRODUCAO = [
    "A esteganografia oculta a própria existência de uma comunicação dentro de uma mídia de "
    "cobertura, diferindo da criptografia, que torna a mensagem ininteligível mas assume sua "
    "presença; a esteganálise é a contraparte, decidir se uma mídia carrega conteúdo oculto "
    "(Fridrich, 2009). Com o aprendizado profundo, ambas passaram a ser formuladas como um jogo "
    "entre redes neurais, no qual um gerador aprende a esconder e um detector a encontrar — "
    "formulação consolidada pelo HiDDeN (Zhu et al., 2018) e pelo SteganoGAN (Zhang et al., "
    "2019).",

    "Nessa formulação competem capacidade, imperceptibilidade e indetectabilidade estatística, e "
    "otimizar as três de uma vez produz instabilidade de convergência e impede atribuir um "
    "resultado ruim a um componente específico. O problema investigado é quanto cada componente "
    "adversarial contribui para a indetectabilidade, e a que custo em qualidade. O objetivo é "
    "construir e avaliar um modelo de esteganografia por GAN em camadas, acrescentando um "
    "componente por vez com critérios declarados antes de cada etapa.",

    "A justificativa é metodológica: é comum reportar a taxa de erro do detector que participou "
    "do próprio treino, número que mede aquele adversário e não a segurança do método. O "
    "trabalho dá continuidade ao STEGS2GAN (Souza; Correia; Trevisani, 2026), incorporando o "
    "treinamento por Wasserstein — ali listado como proposta futura — e tratando a avaliação "
    "independente como etapa obrigatória.",
]

METODOLOGIA = [
    "Os experimentos usam o BOSSBase 1.01 (Bas; Filler; Pevný, 2011), padrão da área: 10.000 "
    "imagens em tons de cinza redimensionadas para 128×128 pixels, sendo 9.500 para treino e "
    "500 para validação. A carga é fixa em 1.024 bits por imagem, ou 0,0625 bits por pixel: uma "
    "mensagem binária de 32×32 é replicada espacialmente até a resolução da imagem, e o "
    "decodificador agrega as cópias redundantes pela média.",

    "O codificador é uma rede convolucional densa que recebe imagem e mensagem e produz um "
    "resíduo somado à imagem original; o decodificador recupera os bits a partir do resultado. "
    "A perda combina erro quadrático médio sobre a imagem e entropia cruzada binária sobre os "
    "bits. Sobre essa base, cada fase acrescenta um componente e fixa seus critérios de saída "
    "antes do treinamento. A Fase 2, já executada, insere um crítico WGAN-GP (Arjovsky; Chintala; "
    "Bottou, 2017; Gulrajani et al., 2017) com penalidade de gradiente e cinco passos de crítico "
    "por passo do gerador. A Fase 2.5, também executada, troca o codificador e o decodificador "
    "densos por blocos residuais no modelo do STEGS2GAN (Souza; Correia; Trevisani, 2026). A "
    "fase seguinte, ainda a executar, insere um esteganalisador SRNet (Boroumand; Chen; "
    "Fridrich, 2019) no laço adversarial, exigindo AUC próxima de 0,5 sem perder mais de 1 dB "
    "de PSNR. Cada fase parte dos pesos da anterior, "
    "exceto a troca de arquitetura, que não admite warm start. As métricas são acurácia de bits, "
    "PSNR e SSIM "
    "(Wang et al., 2004) para qualidade, e a área sob a curva ROC (AUC) do esteganalisador para detecção, calculada "
    "com imagens pareadas, em GPU no Google Colab. A Figura 1 resume a arquitetura resultante "
    "ao fim da Fase 2.5.",
]

# 5 colunas cabem nos 9,25 cm da coluna do modelo; a contagem de épocas foi para a
# legenda e a AUC para o texto, porque 7 colunas estouravam a largura.
TABELA_1 = [
    ["Fase", "Acurácia", "PSNR (dB)", "SSIM", "Épocas ≥ 35 dB"],
    ["0/1 — sem adversário", "0,9999", "35,16", "0,963", "1 de 120"],
    ["2 — WGAN-GP", "0,9997", "36,30", "0,974", "56 de 60"],
    ["2.5 — residual", "0,9998", "38,27", "0,980", "60 de 60"],
]
LARGURAS_1 = [1644, 900, 900, 700, 1100]
LEGENDA_1 = ("Tabela 1. Resultados das fases já executadas (BOSSBase, 128×128, 0,0625 bpp; "
             "120 épocas na Fase 0/1 e 60 nas Fases 2 e 2.5).")

FIGURA_ARQ = "figuras/arquitetura_fase25.png"
LEGENDA_FIG_ARQ = ("Figura 1. Arquitetura do modelo até a Fase 2.5. Em laranja, o que a "
                   "Fase 2 acrescentou; em azul, os blocos cujo miolo a Fase 2.5 trocou.")

FIGURA_CURVA = "figuras/curva_denso_residual.png"
LEGENDA_FIG_CURVA = ("Figura 2. PSNR e SSIM dos geradores denso e residual ao longo das "
                     "Fases 0 e 2, em linha do tempo contínua.")
FONTE = "Fonte: Elaborada pelo autor (2026)."

RESULTADOS_ANTES = ["A Tabela 1 apresenta as três fases já executadas."]

RESULTADOS_MEIO = [
    "A recuperação da mensagem foi praticamente perfeita em todas as fases, o que desloca a "
    "discussão para a qualidade da imagem. O ganho da Fase 2 não está no pico, e sim na "
    "estabilidade: sem o crítico, o PSNR alcançou 35 dB em uma única das 120 épocas, com média "
    "de 33,87 dB nas dez últimas; com o crítico, 56 das 60 superaram o limiar, com média de "
    "36,15 dB.",

    "A Fase 2.5 trocou a topologia densa do gerador por blocos residuais e produziu o maior "
    "ganho isolado até aqui: 38,27 dB contra 36,30 dB, sem custo de acurácia, com as 60 épocas "
    "acima do melhor ponto do denso e a perturbação máxima por pixel caindo de 0,388 para "
    "0,203 — o que é favorável à indetectabilidade. A topologia residual tem, porém, 2,13 vezes "
    "mais parâmetros a igual largura de camada: o ganho ainda não separa topologia de "
    "capacidade, e a ablação a igual número de parâmetros está prevista. A Figura 2 põe as duas "
    "arquiteturas na mesma linha do tempo: o gerador residual está acima do denso desde a Fase "
    "0 e mantém a distância depois que o crítico entra.",
]

RESULTADOS_DEPOIS = [
    "Como resultados esperados, pretende-se: (i) inserir um esteganalisador no laço "
    "adversarial e medir quanto de qualidade de imagem custa a indetectabilidade; (ii) treinar "
    "um esteganalisador independente sobre imagens não vistas pelo gerador e reportar a "
    "probabilidade de erro P_E, que mede segurança e não o desempenho de um adversário "
    "específico; (iii) levantar a curva segurança × capacidade de 0,0156 a 1,0 bpp; e (iv) "
    "avaliar a robustez a recompressão JPEG por camadas de ruído diferenciáveis (Zhu et al., "
    "2018).",
]

CONCLUSOES = [
    "O protocolo em camadas permite atribuir cada resultado ao componente que o causou. O "
    "crítico WGAN-GP mostrou-se benéfico e sobretudo estabilizador: elevou o PSNR de 35,16 dB "
    "para 36,30 dB sem custo de acurácia e, mais importante, transformou um limiar atingido uma "
    "única vez em 120 épocas em um patamar sustentado por 56 de 60. A troca por blocos "
    "residuais somou outros 1,97 dB, ainda sem separar topologia de capacidade. A etapa "
    "seguinte insere o "
    "esteganalisador no laço adversarial. A decisão metodológica que orienta o restante do "
    "trabalho é que a taxa de erro do detector que participou do treino não é evidência de "
    "segurança: por isso a avaliação com um esteganalisador independente entrou no cronograma "
    "como etapa obrigatória, e não como complemento.",
]

AGRADECIMENTOS = ("Os autores agradecem ao IFSP – Câmpus Presidente Epitácio pela infraestrutura "
                  "e suporte fornecidos.")

# (antes do título, título em negrito, depois do título)
REFERENCIAS = [
    ("ARJOVSKY, M.; CHINTALA, S.; BOTTOU, L. ",
     "Wasserstein generative adversarial networks",
     ". In: International Conference on Machine Learning (ICML). PMLR, 2017. p. 214-223. "
     "Disponível em: https://proceedings.mlr.press/v70/arjovsky17a.html."),
    ("BAS, P.; FILLER, T.; PEVNÝ, T. ",
     "“Break our steganographic system”: the ins and outs of organizing BOSS",
     ". In: International Workshop on Information Hiding. Praga: Springer, 2011. p. 59-70. "
     "Disponível em: [completar]."),
    ("BOROUMAND, M.; CHEN, M.; FRIDRICH, J. ",
     "Deep residual network for steganalysis of digital images",
     ". IEEE Transactions on Information Forensics and Security, v. 14, n. 5, p. 1181-1193, "
     "2019. Disponível em: [completar]."),
    ("FRIDRICH, J. ",
     "Steganography in Digital Media: Principles, Algorithms, and Applications",
     ". Cambridge: Cambridge University Press, 2009. Disponível em: "
     "https://books.google.com/books/about/Steganography_in_Digital_Media.html?id=wcAZ-QEthqkC."),
    ("GULRAJANI, I.; AHMED, F.; ARJOVSKY, M.; DUMOULIN, V.; COURVILLE, A. ",
     "Improved training of Wasserstein GANs",
     ". Advances in Neural Information Processing Systems (NIPS), v. 30, p. 5767-5777, 2017. "
     "Disponível em: https://arxiv.org/abs/1704.00028."),
    ("SOUZA, J. P. B. S.; CORREIA, B. W. B.; TREVISANI, K. M. ",
     "STEGS2GAN - um modelo de aprendizado profundo para esteganografia e esteganálise em "
     "imagens digitais utilizando redes adversárias generativas",
     ". In: Workshop de Trabalhos de Iniciação Científica e de Graduação (WTICG). Porto Alegre: "
     "SBC, 2026. Disponível em: [completar]."),
    ("WANG, Z.; BOVIK, A. C.; SHEIKH, H. R.; SIMONCELLI, E. P. ",
     "Image quality assessment: from error visibility to structural similarity",
     ". IEEE Transactions on Image Processing, v. 13, n. 4, p. 600-612, 2004. Disponível em: "
     "https://doi.org/10.1109/TIP.2003.819861."),
    ("ZHANG, K. A.; CUESTA-INFANTE, A.; XU, L.; VEERAMACHANENI, K. ",
     "SteganoGAN: High Capacity Image Steganography with GANs",
     ". arXiv preprint arXiv:1901.03892, 2019. Disponível em: https://arxiv.org/abs/1901.03892."),
    ("ZHU, J.; KAPLAN, R.; JOHNSON, J.; FEI-FEI, L. ",
     "HiDDeN: Hiding Data With Deep Networks",
     ". In: Proceedings of the European Conference on Computer Vision (ECCV). Munique, Alemanha: "
     "Springer, 2018. p. 657-672. Disponível em: https://arxiv.org/abs/1807.09937."),
]

# --------------------------------------------------------------------------- utilidades


def texto(paragrafo, novo):
    """Troca o texto mantendo o primeiro run (fonte, tamanho, itálico vêm dele)."""
    runs = paragrafo.runs
    if not runs:
        paragrafo.add_run(novo)
        return paragrafo
    runs[0].text = novo
    for r in runs[1:]:
        r._element.getparent().remove(r._element)
    return paragrafo


def apagar(paragrafo):
    paragrafo._element.getparent().remove(paragrafo._element)


def clonar_antes(prototipo, ancora, novo_texto):
    """Copia um parágrafo já formatado e o insere antes da âncora, com outro texto."""
    elem = copy.deepcopy(prototipo._element)
    ancora._element.addprevious(elem)
    p = docx.text.paragraph.Paragraph(elem, ancora._parent)
    return texto(p, novo_texto)


def montar_tabela(doc, ancora, linhas, larguras, pt=8):
    tabela = doc.add_table(rows=len(linhas), cols=len(linhas[0]))
    tabela.style = doc.styles["Table Grid"]

    # Largura fixa: sem isto o Word/LibreOffice estica a tabela além da coluna e corta
    # as últimas colunas na exportação.
    tabela.autofit = False
    prop = tabela._tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    prop.append(layout)
    largura = OxmlElement("w:tblW")
    largura.set(qn("w:w"), str(sum(larguras)))
    largura.set(qn("w:type"), "dxa")
    prop.append(largura)

    for i, linha in enumerate(linhas):
        for j, valor in enumerate(linha):
            celula = tabela.cell(i, j)
            celula.width = Twips(larguras[j])
            celula.text = ""
            p = celula.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(valor)
            run.font.name = "Times New Roman"
            run.font.size = Pt(pt)
            run.bold = (i == 0)
    ancora._element.addprevious(tabela._tbl)
    return tabela


def montar_figura(doc, ancora, caminho, largura_cm=8.8):
    """Insere a imagem antes da âncora, centralizada e dentro da largura da coluna."""
    if not os.path.exists(os.path.join(RAIZ, caminho)):
        print(f"  [figura] {caminho} não encontrada — pulando")
        return None
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.add_run().add_picture(os.path.join(RAIZ, caminho), width=Cm(largura_cm))
    ancora._element.addprevious(par._element)
    return par


def escrever_md():
    """Gera o rascunho em Markdown a partir das MESMAS constantes deste arquivo.

    Antes o .md era editado à mão e divergia do .docx a cada revisão.
    """
    destino = os.path.join(AQUI, "resumo_mc2t_2026.md")
    linhas = [
        "# Resumo expandido — MC²T 2026",
        "",
        "> **Gerado por `gerar_resumo_docx.py --md`.** Não edite este arquivo à mão: mude as",
        "> constantes do script e rode de novo, senão ele volta a divergir do `.docx`.",
        "> O documento pronto é `resumo_mc2t_2026.docx`; renderizado, fecha em 2 páginas.",
        "",
        "---", "",
        "## Título", "", f"**{TITULO}**", "",
        "## Autores", "", AUTORES, "", f"{AFILIACAO_1}", "", f"{AFILIACAO_2}", "", EMAILS, "",
        "## Resumo", "", f"*{RESUMO}*", "",
        f"**{PALAVRAS_CHAVE}**", "",
        "## Introdução", "",
    ]
    linhas += [par + "\n" for par in INTRODUCAO]
    linhas += ["## Metodologia", ""] + [par + "\n" for par in METODOLOGIA]
    linhas += [f"![{LEGENDA_FIG_ARQ}]({FIGURA_ARQ})", "",
               f"**{LEGENDA_FIG_ARQ}** {FONTE}", ""]
    linhas += ["## Resultados parciais e esperados", ""] + [par + "\n" for par in RESULTADOS_ANTES]
    linhas += [f"**{LEGENDA_1}**", ""]
    linhas += ["| " + " | ".join(TABELA_1[0]) + " |",
               "|" + "---|" * len(TABELA_1[0])]
    linhas += ["| " + " | ".join(l) + " |" for l in TABELA_1[1:]]
    linhas += ["", FONTE, ""] + [par + "\n" for par in RESULTADOS_MEIO]
    linhas += [f"![{LEGENDA_FIG_CURVA}]({FIGURA_CURVA})", "",
           f"**{LEGENDA_FIG_CURVA}** {FONTE}", ""]
    linhas += [par + "\n" for par in RESULTADOS_DEPOIS]
    linhas += ["## Conclusões", ""] + [par + "\n" for par in CONCLUSOES]
    linhas += ["## Agradecimentos", "", AGRADECIMENTOS, "", "## Referências", ""]
    linhas += [f"{a}**{b}**{c}\n" for a, b, c in REFERENCIAS]
    with open(destino, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))
    print(f"gerado: {destino}")


def main():
    modelo = sys.argv[1] if len(sys.argv) > 1 else MODELO_PADRAO
    if not os.path.exists(modelo):
        sys.exit(f"modelo não encontrado: {modelo}")
    shutil.copy(modelo, SAIDA)
    doc = docx.Document(SAIDA)
    p = doc.paragraphs

    # --- título (vive numa caixa de texto, duplicada por compatibilidade) ---
    trocados = 0
    for cx in doc.element.body.iter(W + "txbxContent"):
        for par in cx.iter(W + "p"):
            corrido = "".join(t.text or "" for t in par.iter(W + "t"))
            if corrido.startswith("Manipulador robótico"):
                ts = list(par.iter(W + "t"))
                ts[0].text = TITULO
                for t in ts[1:]:
                    t.text = ""
                trocados += 1
            elif corrido.startswith("Deve ser redigido em Times New Roman, 12"):
                for t in par.iter(W + "t"):
                    t.text = ""

    # --- cabeçalho e autoria ---
    texto(p[3], AREA)
    texto(p[5], AUTORES)
    texto(p[7], AFILIACAO_1)
    texto(p[8], AFILIACAO_2)
    texto(p[9], EMAILS)
    apagar(p[10])                      # nota do modelo sobre autoria na versão final

    texto(p[14], RESUMO)
    texto(p[16], PALAVRAS_CHAVE)

    # --- introdução (o modelo traz 2 parágrafos; o 3º é clonado) ---
    texto(p[18], INTRODUCAO[0])
    texto(p[19], INTRODUCAO[1])
    clonar_antes(p[18], p[20], INTRODUCAO[2])

    # --- metodologia ---
    texto(p[21], METODOLOGIA[0])
    texto(p[22], METODOLOGIA[1])

    prot_corpo, prot_legenda, prot_fonte, prot_vazio = p[18], p[28], p[29], p[27]
    prot_corpo = copy.deepcopy(prot_corpo._element)
    prot_legenda_e = copy.deepcopy(prot_legenda._element)
    prot_fonte_e = copy.deepcopy(prot_fonte._element)
    prot_vazio_e = copy.deepcopy(prot_vazio._element)

    class _Prot:
        def __init__(self, elem, pai):
            self._element = elem
            self._parent = pai
    pai = p[18]._parent
    P_CORPO = _Prot(prot_corpo, pai)
    P_LEG = _Prot(prot_legenda_e, pai)
    P_FONTE = _Prot(prot_fonte_e, pai)
    P_VAZIO = _Prot(prot_vazio_e, pai)

    # o modelo põe a Tabela 1 e a Figura 1 dentro da Metodologia; aqui elas saem de lá
    for i in range(23, 36):
        apagar(p[i])
    for t in list(doc.tables):
        t._tbl.getparent().remove(t._tbl)

    # --- figura da arquitetura, no fim da Metodologia ---
    ancora_metodologia = p[36]
    clonar_antes(P_VAZIO, ancora_metodologia, "")
    montar_figura(doc, ancora_metodologia, FIGURA_ARQ, largura_cm=7.4)
    clonar_antes(P_LEG, ancora_metodologia, LEGENDA_FIG_ARQ)
    clonar_antes(P_FONTE, ancora_metodologia, FONTE)
    clonar_antes(P_VAZIO, ancora_metodologia, "")

    # --- resultados: tudo reconstruído antes do título "Conclusões" ---
    texto(p[36], "Resultados parciais e esperados")
    for i in range(37, 46):
        apagar(p[i])

    ancora = p[46]                     # parágrafo "Conclusões"
    for t in RESULTADOS_ANTES:
        clonar_antes(P_CORPO, ancora, t)
    clonar_antes(P_LEG, ancora, LEGENDA_1)
    montar_tabela(doc, ancora, TABELA_1, LARGURAS_1)
    clonar_antes(P_FONTE, ancora, FONTE)
    clonar_antes(P_VAZIO, ancora, "")
    for t in RESULTADOS_MEIO:
        clonar_antes(P_CORPO, ancora, t)
    clonar_antes(P_VAZIO, ancora, "")
    montar_figura(doc, ancora, FIGURA_CURVA)
    clonar_antes(P_LEG, ancora, LEGENDA_FIG_CURVA)
    clonar_antes(P_FONTE, ancora, FONTE)
    clonar_antes(P_VAZIO, ancora, "")
    for t in RESULTADOS_DEPOIS:
        clonar_antes(P_CORPO, ancora, t)

    # --- conclusões, agradecimentos ---
    texto(p[47], CONCLUSOES[0])
    texto(p[48], "Agradecimentos")
    texto(p[49], AGRADECIMENTOS)

    # --- referências: 6 do modelo, 6 nossas ---
    apagar(p[51])                      # instrução do modelo
    # o modelo traz 6 vagas de referência; as excedentes são clonadas da última,
    # preservando o espaçamento simples entre entradas que a NBR 6023 pede.
    alvos = [p[53], p[55], p[57], p[59], p[61], p[63]]
    fim_lista = p[64]
    prot_ref = copy.deepcopy(p[63]._element)
    prot_branco = copy.deepcopy(p[62]._element)
    while len(alvos) < len(REFERENCIAS):
        fim_lista._element.addprevious(copy.deepcopy(prot_branco))
        elem = copy.deepcopy(prot_ref)
        fim_lista._element.addprevious(elem)
        alvos.append(docx.text.paragraph.Paragraph(elem, fim_lista._parent))
    for par, (antes, titulo_obra, depois) in zip(alvos, REFERENCIAS):
        # o parágrafo do modelo vem quebrado em vários runs: some com todos menos o
        # primeiro ANTES de acrescentar os novos, senão sobra texto do exemplo no meio.
        base = par.runs[0] if par.runs else par.add_run("")
        for r in par.runs[1:]:
            r._element.getparent().remove(r._element)
        base.text = antes
        base.bold = False
        for trecho, negrito in ((titulo_obra, True), (depois, False)):
            r = par.add_run(trecho)
            r.bold = negrito
            r.font.name = base.font.name
            r.font.size = base.font.size

    doc.save(SAIDA)
    print(f"gerado: {SAIDA}")
    print(f"  título trocado em {trocados} caixa(s) de texto")
    print(f"  parágrafos: {len(docx.Document(SAIDA).paragraphs)} | tabelas: {len(docx.Document(SAIDA).tables)}")


if __name__ == "__main__":
    if "--md" in sys.argv:
        sys.argv.remove("--md")
        escrever_md()
    main()
