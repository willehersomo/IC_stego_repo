#!/usr/bin/env python3
"""
Converte os logs de treino (.log, salvos do Colab) em CSVs de métricas.

Motivo: as Fases 0 e 2 e a varredura de lambda foram rodadas antes de o código passar
a gravar `metricas_treino.csv` / `metricas_val.csv`. O registro daquelas runs existe
só no texto do log. Este script recupera esses números para um formato tabular, sem
precisar re-treinar nada.

Uso:  python3 logs_para_csv.py            # varre os logs conhecidos e grava os CSVs
      python3 logs_para_csv.py --listar   # só mostra o que seria gerado

Normalização aplicada (os logs mudaram de nome entre versões do código):
    bit_acc     -> acuracia_bits
    W           -> w_dist
    psnr=34.5dB -> 34.5   (o sufixo "dB" é removido)
"""
import argparse
import csv
import json
import os
import re

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raiz do repositório

# Notebooks cuja SAÍDA das células é o único registro de um treino. Os .log abaixo são
# gerados a partir deles; se o .log já existir e for igual, nada muda.
# (notebook, log a gerar)
NOTEBOOKS = [
    ("notebooks/fase2_colab.ipynb", "runs/phase2_bossbase/train_colab.log"),
]

# (log de entrada, pasta de saída, rótulo do run)
FONTES = [
    ("runs/phase0_bossbase/train_colab.log", "runs/phase0_bossbase", "phase0_bossbase"),
    ("runs/logs_brutos/phase0_bossbase.log", "runs/phase0_local_cpu", "phase0_local_cpu"),
    ("runs/phase2_bossbase/train_colab.log", "runs/phase2_bossbase",  "phase2_bossbase"),
    ("runs/logs_brutos/sweep_lambda.log",    "runs/sweep",            "sweep_lambda"),
]

def extrair_ipynb(caminho):
    """Concatena a saída de texto de todas as células de código de um notebook.

    O Colab guarda a saída em 'text' (stream) ou em data['text/plain'] (execute_result),
    e cada uma pode vir como string ou lista de linhas.
    """
    with open(caminho, encoding="utf-8") as f:
        nb = json.load(f)
    partes = []
    for celula in nb.get("cells", []):
        if celula.get("cell_type") != "code":
            continue
        for saida in celula.get("outputs", []):
            texto = saida.get("text") or (saida.get("data") or {}).get("text/plain") or ""
            partes.append("".join(texto) if isinstance(texto, list) else str(texto))
    return "\n".join(partes)


RE_EPOCA = re.compile(r"\[\s*(treino|val)\s*\]\s+época\s+(\d+)\s+(.*)$")
RE_PAR = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([+-]?\d+(?:\.\d+)?)")
RE_BRACO = re.compile(r"=+\s*lambda_image\s*=\s*([\d.]+)\s*=+")

RENOMEAR = {"bit_acc": "acuracia_bits", "W": "w_dist"}


def normalizar(chave):
    return RENOMEAR.get(chave, chave)


def ler_log(caminho):
    """Devolve (linhas_treino, linhas_val). Cada linha é um dict com 'epoca' + métricas."""
    treino, val = [], []
    braco = None
    with open(caminho, encoding="utf-8", errors="replace") as f:
        for linha in f:
            m_braco = RE_BRACO.search(linha)
            if m_braco:
                # a varredura repete as épocas por braço; o lambda vira coluna
                braco = m_braco.group(1)
                continue
            m = RE_EPOCA.search(linha)
            if not m:
                continue
            tipo, epoca, resto = m.group(1), int(m.group(2)), m.group(3)
            registro = {"epoca": epoca}
            if braco is not None:
                registro["lambda_imagem"] = braco
            for chave, valor in RE_PAR.findall(resto):
                # guarda o TEXTO do log, não float(): float("36.30") viraria "36.3" no CSV
                # e deixaria de bater com o número citado no relatório e no ROADMAP.
                registro[normalizar(chave)] = valor
            (treino if tipo == "treino" else val).append(registro)
    return treino, val


def gravar(linhas, destino, ordem_inicial=("epoca", "lambda_imagem")):
    """Grava o CSV com a união das colunas encontradas, sem inventar valor faltante."""
    if not linhas:
        return None
    colunas = [c for c in ordem_inicial if any(c in l for l in linhas)]
    for l in linhas:
        for c in l:
            if c not in colunas:
                colunas.append(c)
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    with open(destino, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=colunas, restval="")
        escritor.writeheader()
        escritor.writerows(linhas)
    return destino, len(linhas), colunas


# Runs cujas métricas de validação existem em CSV (geradas aqui ou pelo próprio treino).
RESUMO_FONTES = [
    ("Fase 0 (BOSSBase, GPU)",      "runs/phase0_bossbase/metricas_val.csv"),
    ("Fase 0 (local, CPU, 2k img)", "runs/phase0_local_cpu/metricas_val.csv"),
    ("Fase 2 (WGAN-GP)",            "runs/phase2_bossbase/metricas_val.csv"),
    ("Fase 2.5 (Fase 0, residual)", "runs/phase0_residual/metricas_val.csv"),
    ("Fase 2.5 (Fase 2, residual)", "runs/phase2_residual/metricas_val.csv"),
    ("Fase 3 (SRNet no laço)",      "runs/phase3_bossbase/metricas_val.csv"),
    ("Fase 3 (sanidade, 3 ép.)",    "runs/phase3_sanidade/metricas_val.csv"),
]


def resumir():
    """Uma linha por run: melhor época por PSNR, e a estabilidade nas 10 últimas épocas.

    A média das 10 últimas existe porque "melhor época" esconde pico isolado: a Fase 0
    bateu PSNR >= 35 dB em UMA das 120 épocas. Reportar só o pico seria enganoso.
    """
    linhas = []
    for rotulo, rel in RESUMO_FONTES:
        caminho = os.path.join(RAIZ, rel)
        if not os.path.exists(caminho):
            continue
        with open(caminho, encoding="utf-8") as f:
            dados = [r for r in csv.DictReader(f)]
        if not dados:
            continue
        num = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else None
        com_psnr = [r for r in dados if num(r, "psnr") is not None]
        # Desempate pela ÚLTIMA época, para casar com o ckpt_best.pt: o treino salva
        # a cada melhoria com ">=", então numa sequência empatada fica a mais recente.
        # A Fase 2 empata em 36,30 dB nas épocas 37, 54 e 55 — o checkpoint é o da 55.
        melhor = dados[-1]
        for r in com_psnr:
            if melhor.get("psnr") in (None, "") or num(r, "psnr") >= num(melhor, "psnr"):
                melhor = r
        ult = [r for r in dados[-10:] if num(r, "psnr") is not None]
        media = lambda k, rs: (sum(num(r, k) for r in rs) / len(rs)) if rs else ""
        linhas.append({
            "run": rotulo,
            "epocas": len(dados),
            "melhor_epoca": melhor["epoca"],
            "acuracia_bits": melhor.get("acuracia_bits", ""),
            "psnr": melhor.get("psnr", ""),
            "ssim": melhor.get("ssim", ""),
            "auc_srnet": melhor.get("auc_srnet", ""),
            "psnr_medio_10_ultimas": round(media("psnr", ult), 2) if ult else "",
            "epocas_psnr_maior_igual_35": sum(1 for r in com_psnr if num(r, "psnr") >= 35),
            "fonte": rel,
        })
    return linhas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listar", action="store_true", help="não grava, só mostra")
    args = ap.parse_args()

    for nb_rel, log_rel in NOTEBOOKS:
        nb = os.path.join(RAIZ, nb_rel)
        if not os.path.exists(nb):
            print(f"[pular] {nb_rel} não existe")
            continue
        texto = extrair_ipynb(nb)
        if not texto.strip():
            print(f"[pular] {nb_rel} está sem outputs salvos")
            continue
        destino = os.path.join(RAIZ, log_rel)
        if args.listar:
            print(f"[ipynb] {nb_rel} -> {log_rel} ({len(texto.splitlines())} linhas)")
            continue
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        with open(destino, "w", encoding="utf-8") as f:
            f.write(texto)
        print(f"[ipynb] {nb_rel} -> {log_rel} ({len(texto.splitlines())} linhas)")

    for log_rel, saida_rel, rotulo in FONTES:
        log = os.path.join(RAIZ, log_rel)
        if not os.path.exists(log):
            print(f"[pular] {log_rel} não existe")
            continue
        treino, val = ler_log(log)
        print(f"\n### {rotulo}  ({log_rel})")
        for linhas, nome in ((treino, "metricas_treino.csv"), (val, "metricas_val.csv")):
            if not linhas:
                print(f"    {nome:<22} nenhuma linha encontrada")
                continue
            destino = os.path.join(RAIZ, saida_rel, nome)
            if args.listar:
                print(f"    {nome:<22} {len(linhas):>4} linhas  colunas={sorted(linhas[0])}")
                continue
            _, n, colunas = gravar(linhas, destino)
            print(f"    {nome:<22} {n:>4} linhas -> {os.path.relpath(destino, RAIZ)}")
            print(f"    {'':<22}      colunas: {', '.join(colunas)}")

    linhas = resumir()
    if linhas and not args.listar:
        destino = os.path.join(RAIZ, "runs/resumo_fases.csv")
        gravar(linhas, destino, ordem_inicial=("run", "epocas", "melhor_epoca"))
        print(f"\n### resumo consolidado -> {os.path.relpath(destino, RAIZ)} ({len(linhas)} runs)")
        for l in linhas:
            print(f"    {l['run']:<28} ép {l['melhor_epoca']:>3}/{l['epocas']:<3} "
                  f"psnr {l['psnr']:>5}  (méd 10 últimas {l['psnr_medio_10_ultimas']})  "
                  f"acc {l['acuracia_bits']}")


if __name__ == "__main__":
    main()
