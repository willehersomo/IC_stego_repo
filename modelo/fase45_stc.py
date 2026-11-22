import ctypes
import os
import shutil
import time
from multiprocessing import TimeoutError, get_context

import numpy as np
import torch

import fase30_custo_wgan as f30
import fase40_pe as f40


TETO_CUSTO = 1e7
BITS_CABECALHO = 0
MOLHADO_STC = 2147483647.0
TENTATIVAS = 10
PASSO_CHAVE = 1000003
ESPERA_MAX = 600
LIMITE_SUBSTITUIDAS = 0.01


def bytes_da_mensagem(bpp, amostras):
    return int(bpp * amostras) // 8


def senha_de(semente, i):
    return int(semente) * 100003 + int(i)


@torch.no_grad()
def custos(metodo, x255, ger):
    x = x255.double()
    if metodo["base"] == "hill":
        rp, rm = f30.custo_hill(x)
    else:
        with f30.p_hill_em(torch.float64):
            rho = f30.custo(f30.mapa_p(ger, x.float()))
        rho = torch.where((x <= 0) | (x >= 255), torch.full_like(rho, f30.CUSTO_MOLHADO), rho)
        rp, rm = rho, rho
    return rp.clamp_max(TETO_CUSTO), rm.clamp_max(TETO_CUSTO)


@torch.no_grad()
def distorcao_simulada(rp, rm, bits):
    pp, pm = f30.achar_lambda(rp, rm, bits)
    d = (pp * rp + pm * rm).flatten(1).sum(1)
    t = (pp + pm).flatten(1).mean(1)
    return d.cpu().numpy(), t.cpu().numpy()


_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        import pystc
        P = ctypes.POINTER
        lib = pystc.stc
        lib.ic_hide.argtypes = [ctypes.c_uint, P(ctypes.c_int), P(ctypes.c_float), ctypes.c_uint,
                                P(ctypes.c_ubyte), P(ctypes.c_int), P(ctypes.c_uint)]
        lib.ic_hide.restype = ctypes.c_int
        lib.ic_unhide.argtypes = [ctypes.c_uint, P(ctypes.c_int), ctypes.c_uint, ctypes.c_uint, P(ctypes.c_ubyte)]
        lib.ic_unhide.restype = ctypes.c_int
        _LIB = lib
    return _LIB


def _ptr(a, tipo):
    return a.ctypes.data_as(ctypes.POINTER(tipo))


def esconder(capa, rp, rm, bits, chave):
    lib = _lib()
    a = capa.reshape(-1).astype(np.int32)
    n = a.size
    perm = np.random.default_rng(chave).permutation(n)
    c = np.ascontiguousarray(a[perm])
    custo = np.zeros((n, 3), np.float32)
    custo[:, 0] = np.where(c <= 0, MOLHADO_STC, rm.reshape(-1)[perm])
    custo[:, 2] = np.where(c >= 255, MOLHADO_STC, rp.reshape(-1)[perm])
    msg = np.ascontiguousarray(bits, dtype=np.uint8)
    st = np.zeros(n, np.int32)
    nb = np.zeros(2, np.uint32)
    rc = lib.ic_hide(n, _ptr(c, ctypes.c_int), _ptr(custo, ctypes.c_float), msg.size, _ptr(msg, ctypes.c_ubyte),
                     _ptr(st, ctypes.c_int), _ptr(nb, ctypes.c_uint))
    if rc != 0 or int(nb.sum()) != msg.size:
        return None, nb
    saida = np.empty(n, np.int64)
    saida[perm] = st
    return saida.reshape(capa.shape), nb


def extrair(st, chave, nb, nbits):
    lib = _lib()
    a = st.reshape(-1).astype(np.int32)
    n = a.size
    perm = np.random.default_rng(chave).permutation(n)
    s = np.ascontiguousarray(a[perm])
    msg = np.zeros(nbits, np.uint8)
    rc = lib.ic_unhide(n, _ptr(s, ctypes.c_int), int(nb[0]), int(nb[1]), _ptr(msg, ctypes.c_ubyte))
    return msg if rc == 0 else None


def _tarefa(args):
    i, capa, rp, rm, nbytes, senha = args
    bits = np.unpackbits(np.frombuffer(np.random.default_rng(senha).bytes(nbytes), np.uint8))
    t0 = time.time()
    st, ok = None, False
    for k in range(TENTATIVAS):
        chave = senha + k * PASSO_CHAVE
        st, nb = esconder(capa, rp, rm, bits, chave)
        ok = st is not None and np.array_equal(extrair(st, chave, nb, bits.size), bits)
        if ok:
            break
    t_emb = time.time() - t0
    if st is None:
        st = capa.astype(np.int64)
    delta = st - capa.astype(np.int64)
    dist = float(rp[delta == 1].astype(np.float64).sum() + rm[delta == -1].astype(np.float64).sum())
    return (i, st.clip(0, 255).astype(np.uint8), bool(ok), dist, float((delta != 0).mean()),
            int(np.abs(delta).max()), t_emb, k + 1)


def _tarefas(metodo, capas, indices, bpp, cfg, ger, sim, lote):
    dev = cfg["dispositivo"]
    nbytes = bytes_da_mensagem(bpp, capas[0].numel())
    for k in range(0, len(indices), lote):
        idx = indices[k:k + lote]
        x = capas[idx].to(dev)
        rp, rm = custos(metodo, x, ger)
        d, t = distorcao_simulada(rp, rm, nbytes * 8 + BITS_CABECALHO)
        rp32, rm32 = rp.float().cpu().numpy(), rm.float().cpu().numpy()
        cx = capas[idx].numpy()
        for j, i in enumerate(idx):
            sim[int(i)] = (float(d[j]), float(t[j]))
            yield (int(i), cx[j], rp32[j], rm32[j], nbytes, senha_de(cfg["semente_stc"], i))


def embutir_stc(metodo, capas, indices, bpp, cfg, processos=None, log=print):
    cfg = dict(f40.CONFIG, **cfg)
    ger = f30.carregar_gerador(metodo["ckpt"], cfg["dispositivo"]) if metodo["base"] == "custo" else None
    processos = processos or max(1, (os.cpu_count() or 2) - 1)
    lote = f30._lote_por_tamanho(64, capas)
    sim = {}
    estego = capas.clone()
    res = {}
    t0 = time.time()
    with get_context("spawn").Pool(processos) as pool:
        it = pool.imap_unordered(_tarefa, _tarefas(metodo, capas, list(indices), bpp, cfg, ger, sim, lote),
                                 chunksize=1)
        while True:
            try:
                r = it.next(timeout=ESPERA_MAX)
            except StopIteration:
                break
            except TimeoutError as e:
                raise RuntimeError(f"{metodo['nome']} @ {bpp:g}: nenhum resultado do STC em {ESPERA_MAX}s "
                                   f"({len(res)}/{len(indices)} prontas); um processo auxiliar morreu. PARE") from e
            i, st, ok, dist, taxa, dmax, t_emb, tent = r
            estego[i] = torch.from_numpy(st)
            res[i] = (ok, dist, taxa, dmax, t_emb, tent)
            if len(res) % 1000 == 0:
                log(f"[stc   ] {metodo['nome']} @ {bpp:g}: {len(res)}/{len(indices)} em {time.time() - t0:.0f}s")
    ids = sorted(res)
    ok = np.array([res[i][0] for i in ids])
    bons = [i for i in ids if res[i][0]] or ids
    dist = np.array([res[i][1] for i in bons])
    taxa = np.array([res[i][2] for i in bons])
    dmax = max(res[i][3] for i in ids)
    d_sim = np.array([sim[i][0] for i in bons])
    t_sim = np.array([sim[i][1] for i in bons])
    nbytes = bytes_da_mensagem(bpp, capas[0].numel())
    est = {"n_imagens": len(ids), "recuperacao": float(ok.mean()), "falhas": int((~ok).sum()),
           "bytes_por_imagem": nbytes, "bpp_real": (nbytes * 8 + BITS_CABECALHO) / capas[0].numel(),
           "razao_distorcao": float(np.median(dist / np.maximum(d_sim, 1e-12))),
           "razao_distorcao_media": float(np.mean(dist / np.maximum(d_sim, 1e-12))),
           "taxa_stc": float(taxa.mean()), "taxa_simulada": float(t_sim.mean()),
           "delta_max": int(dmax), "com_retentativa": int(sum(res[i][5] > 1 for i in ids)),
           "tentativas_max": int(max(res[i][5] for i in ids)), "seg_por_imagem": float(np.mean([res[i][4] for i in ids])),
           "minutos": (time.time() - t0) / 60}
    return estego, est, [i for i in ids if not res[i][0]]


@torch.no_grad()
def substituir_por_simulado(metodo, capas, estego, ids, bpp, cfg):
    dev = cfg["dispositivo"]
    ger = f30.carregar_gerador(metodo["ckpt"], dev) if metodo["base"] == "custo" else None
    x = capas[list(ids)].to(dev)
    rp, rm = custos(metodo, x, ger)
    g = torch.Generator().manual_seed(int(cfg["semente_stc"]))
    delta = f30.simular_embutimento(x.double(), rp, rm, bytes_da_mensagem(bpp, capas[0].numel()) * 8, g)
    estego[list(ids)] = (x.to(torch.int16) + delta).clamp(0, 255).to(estego.dtype).cpu()


def caminho_cache(nome, bpp, cfg, n):
    return os.path.join(cfg["pasta_saida"], "corpus",
                        f"estego_{nome}_{bpp:g}_{cfg['canais']}x{cfg['tamanho_img']}_{n}.pt")


def gerar_corpus_stc(metodo, capas, idx_te, bpp, cfg, processos=None, log=print):
    cfg = dict(f40.CONFIG, **cfg)
    local = caminho_cache(metodo["nome"], bpp, cfg, len(capas))
    rel = os.path.relpath(local, cfg["pasta_saida"])
    achado = f40._no_espelho_ou_local(rel, cfg)
    if achado and achado != local:
        os.makedirs(os.path.dirname(local), exist_ok=True)
        shutil.copy2(achado, local)
    if os.path.exists(local):
        estego, info = torch.load(local)
        log(f"[stc   ] {metodo['nome']} @ {bpp:g}: reaproveitado (recuperação {info['recuperacao']:.4f})")
        return estego, info
    estego, est, falhas = embutir_stc(metodo, capas, range(len(capas)), bpp, cfg, processos, log)
    if len(falhas) > LIMITE_SUBSTITUIDAS * len(capas):
        raise RuntimeError(f"{metodo['nome']} @ {bpp:g}: {len(falhas)} imagens não recuperaram a mensagem "
                           f"(primeiras: {falhas[:10]}). PARE: o STC não está correto.")
    if falhas:
        substituir_por_simulado(metodo, capas, estego, falhas, bpp, cfg)
        log(f"[stc   ] {metodo['nome']} @ {bpp:g}: STC não comportou a carga em {len(falhas)} imagens "
            f"({len(falhas) / len(capas):.2%}); nelas foi usado o estego simulado com os mesmos custos "
            f"(primeiras: {falhas[:10]})")
    est["substituidas_simulado"] = len(falhas)
    est["ids_substituidas"] = list(map(int, falhas))
    info = dict(f40.metricas_imagem(capas[idx_te], estego[idx_te]), **est)
    os.makedirs(os.path.dirname(local), exist_ok=True)
    torch.save((estego, info), local)
    f40._espelhar_arquivo(local, cfg)
    log(f"[stc   ] {metodo['nome']} @ {bpp:g} bpp: {len(estego)} estegos em {est['minutos']:.1f} min | "
        f"recuperação {est['recuperacao']:.4f} | outra chave em {est['com_retentativa']} | taxa STC {est['taxa_stc']:.2%} × simulada "
        f"{est['taxa_simulada']:.2%} | distorção STC/simulada {est['razao_distorcao']:.3f} | "
        f"teste: PSNR {info['psnr']:.2f}  SSIM {info['ssim']:.5f}")
    return estego, info


def sanidade(metodos, capas, bpp, cfg, n=32, processos=None, log=print):
    rng = np.random.default_rng(0)
    idx = sorted(rng.choice(len(capas), n, replace=False).tolist())
    saida = {}
    for m in metodos:
        _, est, falhas = embutir_stc(m, capas, idx, bpp, cfg, processos, log)
        saida[m["nome"]] = est
        log(f"[sanid.] {m['nome']} @ {bpp:g}: recuperação {est['recuperacao']:.4f} ({est['falhas']} falhas) | "
            f"{est['bytes_por_imagem']} bytes/imagem ({est['bpp_real']:.4f} bpp) | "
            f"taxa STC {est['taxa_stc']:.2%} × simulada {est['taxa_simulada']:.2%} | "
            f"distorção STC/simulada {est['razao_distorcao']:.3f} | |Δ| máx {est['delta_max']} | "
            f"outra chave em {est['com_retentativa']} (máx {est['tentativas_max']} tentativas) | "
            f"{est['seg_por_imagem']:.2f} s/imagem")
    return saida
