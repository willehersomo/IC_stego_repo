import contextlib
import copy
import json
import csv
import glob
import math
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image

import fase0_codificador_decodificador as base
import fase2_wgan_gp as fase2
import fase25_residual as f25
import fase3_esteganalisador as f3



CONFIG_SPLIT = {
    "separador_cena": "__",
    "semente_split": 42,
    "fracao_val": 0.05,
    "n_val_detector": 1000,
    "max_imagens": None,
}

def pe_auc(scores, rotulos):

    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(rotulos, dtype=np.int64)
    ordem = np.argsort(-s, kind="mergesort")
    s, y = s[ordem], y[ordem]
    ultimos = np.r_[np.diff(s) != 0, True]
    vp = np.r_[0, np.cumsum(y)[ultimos]]
    fp = np.r_[0, np.cumsum(1 - y)[ultimos]]
    tpr, fpr = vp / y.sum(), fp / (len(y) - y.sum())
    erro = (fpr + (1.0 - tpr)) / 2.0
    pe = float(np.minimum(erro, 1.0 - erro).min())
    trap = getattr(np, "trapezoid", None) or np.trapz
    auc = float(trap(tpr, fpr))
    return pe, max(auc, 1.0 - auc)


def dividir(cfg):
    caminhos = []
    for ext in base.EXTENSOES_IMG:
        caminhos += glob.glob(os.path.join(cfg["pasta_dados"], "**", "*" + ext), recursive=True)
        caminhos += glob.glob(os.path.join(cfg["pasta_dados"], "**", "*" + ext.upper()), recursive=True)
    caminhos = sorted(set(caminhos))
    sep = cfg.get("separador_cena")
    if sep and caminhos and all(sep in os.path.basename(c) for c in caminhos):
        return _dividir_por_cena(caminhos, sep, cfg)
    random.Random(cfg["semente_split"]).shuffle(caminhos)
    if cfg.get("max_imagens"):
        caminhos = caminhos[:cfg["max_imagens"]]
    n_teste = max(1, int(len(caminhos) * cfg["fracao_val"]))
    n_val = min(cfg["n_val_detector"], (len(caminhos) - n_teste) // 5)
    idx = np.arange(len(caminhos))
    return caminhos, idx[n_teste + n_val:], idx[n_teste:n_teste + n_val], idx[:n_teste]


def _dividir_por_cena(caminhos, sep, cfg):
    grupos = {}
    for c in caminhos:
        grupos.setdefault(os.path.basename(c).split(sep)[0], []).append(c)
    cenas = sorted(grupos)
    random.Random(cfg["semente_split"]).shuffle(cenas)
    por_cena = len(caminhos) / len(cenas)
    if cfg.get("max_imagens"):
        cenas = cenas[:max(3, math.ceil(cfg["max_imagens"] / por_cena))]
    n_t = max(1, int(len(cenas) * cfg["fracao_val"]))
    n_v = min(math.ceil(cfg["n_val_detector"] / por_cena), (len(cenas) - n_t) // 5)
    ordem = [c for k in cenas for c in grupos[k]]
    n_teste = sum(len(grupos[k]) for k in cenas[:n_t])
    n_val = sum(len(grupos[k]) for k in cenas[n_t:n_t + n_v])
    idx = np.arange(len(ordem))
    return ordem, idx[n_teste + n_val:], idx[n_teste:n_teste + n_val], idx[:n_teste]




CUSTO_MOLHADO = 1e10   


def _pad_simetrico(x, p):
    h, w = x.shape[-2:]
    iy = torch.cat([torch.arange(p - 1, -1, -1), torch.arange(h), torch.arange(h - 1, h - 1 - p, -1)])
    ix = torch.cat([torch.arange(p - 1, -1, -1), torch.arange(w), torch.arange(w - 1, w - 1 - p, -1)])
    return x[..., iy.to(x.device), :][..., ix.to(x.device)]


def custo_hill(x, dtype=torch.float64):
    forma = x.shape
    x = x.reshape(-1, 1, *forma[-2:]).to(dtype)
    kb = torch.tensor([[-1., 2., -1.], [2., -4., 2.], [-1., 2., -1.]], dtype=x.dtype,
                      device=x.device).view(1, 1, 3, 3)
    l1 = torch.full((1, 1, 3, 3), 1 / 9, dtype=x.dtype, device=x.device)
    l2 = torch.full((1, 1, 15, 15), 1 / 225, dtype=x.dtype, device=x.device)
    r = F.conv2d(_pad_simetrico(x, 1), kb).abs()
    rho = 1.0 / (F.conv2d(_pad_simetrico(r, 1), l1) + 1e-10)
    rho = F.conv2d(_pad_simetrico(rho, 7), l2).clamp_max(CUSTO_MOLHADO)
    return (torch.where(x >= 255, CUSTO_MOLHADO, rho).reshape(forma),
            torch.where(x <= 0, CUSTO_MOLHADO, rho).reshape(forma))


def _probabilidades(rp, rm, lam):
    ep, em = torch.exp(-lam * rp), torch.exp(-lam * rm)
    z = 1.0 + ep + em
    return ep / z, em / z


def _entropia_bits(pp, pm):
    p0 = 1.0 - pp - pm
    h = -(torch.special.xlogy(pp, pp) + torch.special.xlogy(pm, pm) + torch.special.xlogy(p0, p0))
    return h.flatten(1).sum(1) / math.log(2)


def achar_lambda(rp, rm, bits, iteracoes=60):
    n = rp.shape[0]
    forma = (n,) + (1,) * (rp.dim() - 1)
    lo = torch.full(forma, -40.0, dtype=rp.dtype, device=rp.device)
    hi = torch.full(forma, 40.0, dtype=rp.dtype, device=rp.device)
    for _ in range(iteracoes):
        meio = (lo + hi) / 2
        h = _entropia_bits(*_probabilidades(rp, rm, meio.exp())).view(forma)
        acima = h > bits
        lo, hi = torch.where(acima, meio, lo), torch.where(acima, hi, meio)
    return _probabilidades(rp, rm, ((lo + hi) / 2).exp())


def simular_embutimento(x, rp, rm, bits, gerador_rng, iteracoes=60):

    pp, pm = achar_lambda(rp, rm, bits, iteracoes)
    u = torch.rand(x.shape, generator=gerador_rng, dtype=torch.float64).to(x.device)
    delta = (u < pp).to(torch.int16) - ((u >= pp) & (u < pp + pm)).to(torch.int16)
    return delta



def _lote_por_tamanho(lote, capas):
    return max(1, int(lote * 128 * 128 / capas[0].numel()))



PASSOS = {
    "7a": {"fonte": "div2k", "canais": 1, "tamanho": 128, "por_imagem": 12},
    "7b": {"fonte": "div2k", "canais": 3, "tamanho": 128, "por_imagem": 12},
    "7c": {"fonte": "div2k", "canais": 3, "tamanho": 256, "por_imagem": 6},
    "7d": {"fonte": "coco",  "canais": 3, "tamanho": 128, "por_imagem": 2},
}


def nome_pasta(passo):
    p = PASSOS[passo]
    cor = "cinza" if p["canais"] == 1 else "rgb"
    return f"{p['fonte']}_{cor}{p['tamanho']}"


def recortar(origem, destino, canais, tamanho, por_imagem, semente=42, log=print):
    fotos = sorted(glob.glob(os.path.join(origem, "*.png")) + glob.glob(os.path.join(origem, "*.jpg")))
    if not fotos:
        raise FileNotFoundError(f"nenhuma foto em {origem}")
    os.makedirs(destino, exist_ok=True)
    feito = os.path.join(destino, "_completo.txt")
    if os.path.exists(feito):
        n = len(glob.glob(os.path.join(destino, "*.png")))
        log(f"[recorte] {destino}: {n} recortes já prontos")
        return n

    rng = random.Random(semente)
    t0, n, puladas = time.time(), 0, 0
    for i, foto in enumerate(fotos):
        img = Image.open(foto)
        if canais == 3 and img.mode != "RGB":
            puladas += 1
            continue
        img = img.convert("L" if canais == 1 else "RGB")
        w, h = img.size
        grade = [(x, y) for y in range(0, h - tamanho + 1, tamanho)
                 for x in range(0, w - tamanho + 1, tamanho)]
        if not grade:
            puladas += 1
            continue
        cena = os.path.splitext(os.path.basename(foto))[0]
        for k, (x, y) in enumerate(sorted(rng.sample(grade, min(por_imagem, len(grade))))):
            img.crop((x, y, x + tamanho, y + tamanho)).save(
                os.path.join(destino, f"{cena}__{k:02d}.png"))
            n += 1
        if (i + 1) % 200 == 0:
            log(f"[recorte] {i + 1}/{len(fotos)} fotos, {n} recortes, {time.time() - t0:.0f}s")
    with open(feito, "w") as f:
        f.write(f"{n} recortes de {len(fotos) - puladas} fotos ({puladas} puladas)\n")
    log(f"[recorte] {destino}: {n} recortes de {len(fotos) - puladas} fotos "
        f"({puladas} puladas) em {time.time() - t0:.0f}s")
    return n


def preparar(passo, pasta_fontes, pasta_destino, log=print):
    p = PASSOS[passo]
    origem = os.path.join(pasta_fontes, "DIV2K_train_HR" if p["fonte"] == "div2k" else "val2017")
    destino = os.path.join(pasta_destino, nome_pasta(passo))
    recortar(origem, destino, p["canais"], p["tamanho"], p["por_imagem"], log=log)
    return destino, p




P_MIN = 1e-6


def entropia_ternaria(p):
    p = p.clamp(P_MIN, 0.5 - P_MIN)
    h = -p * torch.log2(p / 2) - (1 - p) * torch.log2(1 - p)
    return h.flatten(1).sum(1)


def p_uniforme(bpp):
    lo, hi = P_MIN, 0.5 - P_MIN
    for _ in range(60):
        meio = (lo + hi) / 2
        if entropia_ternaria(torch.tensor([[[[meio]]]])).item() < bpp:
            lo = meio
        else:
            hi = meio
    return (lo + hi) / 2


P_HILL_DTYPE = torch.float32


@contextlib.contextmanager
def p_hill_em(dtype):
    global P_HILL_DTYPE
    antigo, P_HILL_DTYPE = P_HILL_DTYPE, dtype
    try:
        yield
    finally:
        P_HILL_DTYPE = antigo


@torch.no_grad()
def p_hill(x255, bpp):
    rp, rm = custo_hill(x255, P_HILL_DTYPE)
    pp, pm = achar_lambda(rp, rm, bpp * x255[0].numel())
    return (pp + pm).float().clamp(P_MIN, 0.5 - P_MIN)


@torch.no_grad()
def diferenca_p_hill(x255, bpp):
    with p_hill_em(torch.float32):
        a = p_hill(x255, bpp)
    with p_hill_em(torch.float64):
        b = p_hill(x255, bpp)
    return float((a - b).abs().max())


def double_tanh(p, n, lam):
    return -0.5 * torch.tanh(lam * (p - 2 * n)) + 0.5 * torch.tanh(lam * (p - 2 * (1 - n)))


def molhar(p, x255):
    return torch.where((x255 <= 0) | (x255 >= 255), torch.full_like(p, P_MIN), p)


def custo(p):
    p = p.double().clamp(P_MIN, 0.5 - P_MIN)
    return torch.log(2.0 / p - 2.0).clamp_max(CUSTO_MOLHADO)


class GeradorCusto(nn.Module):

    def __init__(self, canais=1, largura=64, blocos=6, p_inicial=0.05, base_p="uniforme", bpp=0.4):
        super().__init__()
        self.base_p, self.bpp = base_p, bpp
        self.entrada = base.bloco_conv(canais, largura)
        self.residuais = nn.Sequential(*[f25.BlocoResidual(largura) for _ in range(blocos)])
        self.para_p = nn.Conv2d(largura, canais, 3, padding=1)
        with torch.no_grad():
            self.para_p.weight.mul_(0.01)
            if base_p == "hill":
                self.para_p.bias.zero_()
            else:
                q = 2 * p_inicial
                self.para_p.bias.fill_(math.log(q / (1 - q)))

    def forward(self, x):
        z = self.para_p(self.residuais(self.entrada(x - 0.5)))
        self.ultimo_z = z
        if self.base_p == "hill":
            q = (2 * p_hill(torch.round(x * 255.0), self.bpp)).clamp(2 * P_MIN, 1 - 2 * P_MIN)
            z = z + torch.log(q / (1 - q))
        return (0.5 * torch.sigmoid(z)).clamp(P_MIN, 0.5 - P_MIN)




NORMAS = ("batch", "instancia", "afim", "espectral")


class _Afim(nn.Module):

    def __init__(self, canais):
        super().__init__()
        self.peso = nn.Parameter(torch.ones(canais))
        self.vies = nn.Parameter(torch.zeros(canais))
        self.register_buffer("iniciado", torch.tensor(False))

    def forward(self, x):
        if not bool(self.iniciado):
            with torch.no_grad():
                m = x.mean(dim=(0, 2, 3))
                d = x.std(dim=(0, 2, 3)).clamp_min(1e-6)
                self.peso.copy_(1.0 / d); self.vies.copy_(-m / d); self.iniciado.fill_(True)
        return x * self.peso.view(1, -1, 1, 1) + self.vies.view(1, -1, 1, 1)


def _camada_norma(norma, canais):
    if norma == "batch":
        return nn.BatchNorm2d(canais)
    if norma == "instancia":
        return nn.InstanceNorm2d(canais, affine=True)
    if norma in ("afim", "espectral"):
        return _Afim(canais)
    raise ValueError(f"normalização desconhecida: {norma} (use uma de {NORMAS})")


def _trocar_batchnorm(modulo, norma):
    for nome, filho in modulo.named_children():
        if isinstance(filho, nn.BatchNorm2d):
            setattr(modulo, nome, _camada_norma(norma, filho.num_features))
        else:
            _trocar_batchnorm(filho, norma)


def _aplicar_espectral(rede):
    for m in list(rede.modules()):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.utils.parametrizations.spectral_norm(m)


def norma_efetiva(cfg):
    if cfg.get("norma_critico"):
        return cfg["norma_critico"]
    if cfg.get("discriminador", "xunet") == "srnet":
        return "batch"
    return "instancia" if cfg.get("adversario") == "wgan_gp" else "batch"


def _hpf6():
    kv = torch.tensor([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                       [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=torch.float32) / 12
    filtros = [kv]
    kb = torch.zeros(5, 5); kb[1:4, 1:4] = torch.tensor([[-1., 2, -1], [2, -4, 2], [-1, 2, -1]]) / 4
    filtros.append(kb)
    h1 = torch.zeros(5, 5); h1[2, 2], h1[2, 3] = -1, 1
    h2 = torch.zeros(5, 5); h2[2, 1:4] = torch.tensor([1., -2, 1]) / 2
    filtros += [h1, h1.t().clone(), h2, h2.t().clone()]
    return torch.stack(filtros).unsqueeze(1)


class XuNet6HPF(nn.Module):

    def __init__(self, canais=1, norma="batch"):
        super().__init__()
        self.canais = canais
        self.register_buffer("hpf", _hpf6().repeat(canais, 1, 1, 1))
        N = lambda c: _camada_norma(norma, c)
        pool = lambda: nn.AvgPool2d(5, stride=2, padding=2)
        self.g1 = nn.Sequential(nn.Conv2d(6 * canais, 8, 5, padding=2, bias=False))
        self.n1 = N(8)
        self.g2 = nn.Sequential(nn.Conv2d(8, 16, 5, padding=2, bias=False), N(16), nn.Tanh(), pool())
        self.g3 = nn.Sequential(nn.Conv2d(16, 32, 1, bias=False), N(32), nn.ReLU(True), pool())
        self.g4 = nn.Sequential(nn.Conv2d(32, 64, 1, bias=False), N(64), nn.ReLU(True), pool())
        self.g5 = nn.Sequential(nn.Conv2d(64, 128, 1, bias=False), N(128), nn.ReLU(True))
        self.para_logit = nn.Linear(128, 1)
        self.pool1 = pool()
        if norma == "espectral":
            _aplicar_espectral(self)

    def forward(self, x255):
        r = F.conv2d(x255, self.hpf, padding=2, groups=self.canais)
        h = self.pool1(torch.tanh(self.n1(self.g1(r).abs())))
        h = self.g5(self.g4(self.g3(self.g2(h))))
        return self.para_logit(h.mean(dim=(2, 3))).squeeze(1)


class _SRNet255(nn.Module):

    def __init__(self, canais=1, norma="batch"):
        super().__init__()
        self.rede = f3.SRNet(canais, compacta=False)
        if norma != "batch":
            _trocar_batchnorm(self.rede, norma)
        if norma == "espectral":
            _aplicar_espectral(self.rede)

    def forward(self, x255):
        return self.rede(x255 / 255.0)


def construir_discriminador(cfg):
    canais = cfg.get("canais", 1)
    norma = norma_efetiva(cfg)
    if cfg.get("adversario") == "wgan_gp" and norma == "batch":
        print("[aviso ] WGAN-GP com BatchNorm no crítico: a penalidade de gradiente por amostra não "
              "vale. Use norma_critico='instancia' | 'afim' | 'espectral'.")
    if cfg["discriminador"] == "srnet":
        return _SRNet255(canais, norma)
    return XuNet6HPF(canais, norma)




def estego_hill(x255, bpp, rng):
    rp, rm = custo_hill(x255)
    return (x255 + simular_embutimento(x255, rp, rm, bpp * x255[0].numel(), rng)).clamp(0, 255)


def converter_bn_para_afim(modulo):
    for nome, filho in modulo.named_children():
        if isinstance(filho, nn.BatchNorm2d):
            escala = filho.weight.detach() / torch.sqrt(filho.running_var + filho.eps)
            afim = _Afim(filho.num_features).to(filho.weight.device)
            with torch.no_grad():
                afim.peso.copy_(escala)
                afim.vies.copy_(filho.bias.detach() - filho.running_mean * escala)
                afim.iniciado.fill_(True)
            setattr(modulo, nome, afim)
        else:
            converter_bn_para_afim(filho)
    return modulo


@torch.no_grad()
def auc_hill(rede, loader, cfg, bpp=None, semente=4321):
    dev, bpp = cfg["dispositivo"], bpp or cfg["bpp"]
    rng = torch.Generator().manual_seed(semente)
    rede.eval()
    sc, rot = [], []
    for x in loader:
        x255 = torch.round(x.to(dev) * 255.0)
        sc.append(rede(torch.cat([x255, estego_hill(x255, bpp, rng)])).float().cpu())
        rot.append(torch.cat([torch.zeros(len(x)), torch.ones(len(x))]))
    rede.train()
    return pe_auc(torch.cat(sc).numpy(), torch.cat(rot).numpy())[1]


def pretreinar_critico_bce(cfg, saida, epocas=15, lr=1e-4, log=print):
    cfg = dict(CONFIG, **cfg)
    dev = cfg["dispositivo"]
    torch.manual_seed(cfg["semente"])
    loader_tr, loader_va = montar_loaders(cfg)
    rede = _SRNet255(cfg.get("canais", 1), "batch").to(dev)
    opt = torch.optim.Adam(rede.parameters(), lr=lr, betas=(0.5, 0.999))
    rng = torch.Generator().manual_seed(77)
    melhor = (-1.0, None, 0)
    for ep in range(1, epocas + 1):
        t0, soma, n = time.time(), 0.0, 0
        for x in loader_tr:
            x255 = torch.round(x.to(dev) * 255.0)
            with torch.no_grad():
                y = estego_hill(x255, cfg["bpp"], rng)
            s = rede(torch.cat([x255, y]))
            rot = torch.cat([torch.zeros(len(x), device=dev), torch.ones(len(x), device=dev)])
            perda = F.binary_cross_entropy_with_logits(s, rot)
            opt.zero_grad(set_to_none=True); perda.backward(); opt.step()
            soma += perda.item(); n += 1
        auc = auc_hill(rede, loader_va, cfg)
        marca = ""
        if auc > melhor[0]:
            melhor = (auc, {k: v.detach().clone() for k, v in rede.state_dict().items()}, ep)
            marca = "  *"
        log(f"[pré-treino] época {ep:2d}  BCE={soma / max(n, 1):.4f}  AUC val (HILL {cfg['bpp']:g} bpp)="
            f"{auc:.4f}  | {time.time() - t0:.0f}s{marca}")
    rede.load_state_dict(melhor[1])
    auc_bn = auc_hill(rede, loader_va, cfg)
    converter_bn_para_afim(rede)
    auc_afim = auc_hill(rede, loader_va, cfg)
    torch.save(rede.state_dict(), saida)
    log(f"[pré-treino] melhor época {melhor[2]}: AUC {auc_bn:.4f} com BatchNorm → {auc_afim:.4f} já em afim "
        f"(devem ser iguais) -> {saida}")
    return auc_bn, auc_afim


def estego_unif(x255, bpp, rng):
    um = torch.ones(x255.shape, dtype=torch.float64, device=x255.device)
    rp = torch.where(x255 >= 255, CUSTO_MOLHADO, um)
    rm = torch.where(x255 <= 0, CUSTO_MOLHADO, um)
    return (x255 + simular_embutimento(x255, rp, rm, bpp * x255[0].numel(), rng)).clamp(0, 255)


@torch.no_grad()
def auc_ref(rede, loader, cfg, bpp, tipo="hill", semente=4321):
    dev = cfg["dispositivo"]
    rng = torch.Generator().manual_seed(semente)
    fazer = estego_hill if tipo == "hill" else estego_unif
    rede.eval()
    sc, rot = [], []
    for x in loader:
        x255 = torch.round(x.to(dev) * 255.0)
        sc.append(rede(torch.cat([x255, fazer(x255, bpp, rng)])).float().cpu())
        rot.append(torch.cat([torch.zeros(len(x)), torch.ones(len(x))]))
    rede.train()
    return pe_auc(torch.cat(sc).numpy(), torch.cat(rot).numpy())[1]


def pretreinar_critico_curriculo(cfg, pasta, estagios, lr=1e-4, log=print):
    cfg = dict(CONFIG, **cfg)
    dev = cfg["dispositivo"]
    os.makedirs(pasta, exist_ok=True)
    torch.manual_seed(cfg["semente"])
    loader_tr, loader_va = montar_loaders(cfg)
    rede = _SRNet255(cfg.get("canais", 1), "batch").to(dev)
    rng = torch.Generator().manual_seed(77)
    saida = []
    for bpp, epocas in estagios:
        opt = torch.optim.Adam(rede.parameters(), lr=lr, betas=(0.5, 0.999))
        melhor = (-1.0, None, 0)
        for ep in range(1, epocas + 1):
            t0, soma, n = time.time(), 0.0, 0
            for x in loader_tr:
                x255 = torch.round(x.to(dev) * 255.0)
                with torch.no_grad():
                    y = estego_hill(x255, bpp, rng)
                s = rede(torch.cat([x255, y]))
                rot = torch.cat([torch.zeros(len(x), device=dev), torch.ones(len(x), device=dev)])
                perda = F.binary_cross_entropy_with_logits(s, rot)
                opt.zero_grad(set_to_none=True); perda.backward(); opt.step()
                soma += perda.item(); n += 1
            auc = auc_ref(rede, loader_va, cfg, bpp, "hill")
            marca = ""
            if auc > melhor[0]:
                melhor = (auc, {k: v.detach().clone() for k, v in rede.state_dict().items()}, ep)
                marca = "  *"
            log(f"[crítico {bpp:g} bpp] época {ep:2d}  BCE={soma / max(n, 1):.4f}  AUC HILL={auc:.4f}  "
                f"| {time.time() - t0:.0f}s{marca}")
        rede.load_state_dict(melhor[1])
        conv = converter_bn_para_afim(copy.deepcopy(rede))
        r = {"bpp": bpp, "epoca": melhor[2], "auc_hill": auc_ref(conv, loader_va, cfg, bpp, "hill"),
             "auc_unif": auc_ref(conv, loader_va, cfg, bpp, "unif"),
             "arquivo": os.path.join(pasta, f"critico_{bpp:g}.pt")}
        torch.save(conv.state_dict(), r["arquivo"])
        r["arquivo_bn"] = os.path.join(pasta, f"critico_{bpp:g}_bn.pt")
        torch.save(rede.state_dict(), r["arquivo_bn"])
        log(f"[crítico {bpp:g} bpp] melhor época {r['epoca']}: AUC HILL {r['auc_hill']:.4f} | "
            f"AUC ±1 {r['auc_unif']:.4f} (já em afim) -> {r['arquivo']}")
        saida.append(r)
    return saida


def testar_estabilidade(cfg, critico, congelar=(), epocas=6, log=print):
    cfg = dict(CONFIG, **cfg)
    dev, bpp = cfg["dispositivo"], cfg["bpp"]
    torch.manual_seed(cfg["semente"])
    loader_tr, loader_va = montar_loaders(cfg)
    disc = construir_discriminador(dict(cfg, adversario="wgan_gp", norma_critico="afim")).to(dev)
    disc.load_state_dict(torch.load(critico, map_location=dev, weights_only=False))
    cong = tuple(f"rede.{b}." for b in congelar)
    for nome, prm in disc.named_parameters():
        if cong and nome.startswith(cong):
            prm.requires_grad_(False)
    opt = torch.optim.Adam([q for q in disc.parameters() if q.requires_grad], lr=cfg["lr_d"], betas=(0.0, 0.9))
    rng = torch.Generator().manual_seed(55)
    hist = [{"epoca": 0, "auc_hill": auc_ref(disc, loader_va, cfg, bpp, "hill"),
             "auc_unif": auc_ref(disc, loader_va, cfg, bpp, "unif")}]
    log(f"[estabilidade {congelar or 'livre'}] início: AUC HILL {hist[0]['auc_hill']:.4f}  ±1 {hist[0]['auc_unif']:.4f}")
    for ep in range(1, epocas + 1):
        t0 = time.time()
        for x in loader_tr:
            x255 = torch.round(x.to(dev) * 255.0)
            with torch.no_grad():
                y = estego_hill(x255, bpp, rng)
            b = len(x255)
            s = disc(torch.cat([x255, y]))
            perda = s[:b].mean() - s[b:].mean() + cfg["lambda_gp"] * fase2.penalidade_gradiente(disc, x255, y, dev) \
                + cfg.get("lambda_drift", 0.0) * s.pow(2).mean()
            opt.zero_grad(set_to_none=True); perda.backward(); opt.step()
        h = {"epoca": ep, "auc_hill": auc_ref(disc, loader_va, cfg, bpp, "hill"),
             "auc_unif": auc_ref(disc, loader_va, cfg, bpp, "unif")}
        hist.append(h)
        log(f"[estabilidade {congelar or 'livre'}] época {ep}: AUC HILL {h['auc_hill']:.4f}  ±1 {h['auc_unif']:.4f} "
            f"| {time.time() - t0:.0f}s")
    return hist



@torch.no_grad()
def mapa_p(gerador, x255):
    return molhar(gerador(x255.float() / 255.0), x255.float())


@torch.no_grad()
def embutir(gerador, x255, bpp, gerador_rng):
    x255 = x255.to(next(gerador.parameters()).device)
    rho = custo(mapa_p(gerador, x255))
    rho = torch.where((x255 <= 0) | (x255 >= 255), torch.full_like(rho, CUSTO_MOLHADO), rho)
    delta = simular_embutimento(x255, rho, rho, bpp * x255[0].numel(), gerador_rng)
    return (x255.to(torch.int16) + delta).clamp(0, 255).to(torch.uint8)


def carregar_gerador(caminho, dispositivo):
    ck = torch.load(caminho, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    g = GeradorCusto(c.get("canais", 1), c["largura"], c["blocos"], p_uniforme(c["bpp"]),
                     c.get("base", "uniforme"), c["bpp"])
    g.load_state_dict(ck["gerador"])
    return g.to(dispositivo).eval()


@torch.no_grad()
def avaliar_laco_alvo(gerador, disc, loader, cfg):
    dev, b = cfg["dispositivo"], cfg["bpp_laco_alvo"]
    rng = torch.Generator().manual_seed(4321)
    sc = {"braco": [], "hill": [], "unif": []}
    rot = []
    for x in loader:
        x255 = torch.round(x.to(dev) * 255.0)
        bits = b * x255[0].numel()
        with p_hill_em(torch.float64):
            y = {"braco": embutir(gerador, x255, b, rng).float()}
        rp, rm = custo_hill(x255)
        y["hill"] = (x255 + simular_embutimento(x255, rp, rm, bits, rng)).clamp(0, 255)
        um = torch.ones(x255.shape, dtype=torch.float64, device=dev)
        rp = torch.where(x255 >= 255, CUSTO_MOLHADO, um)
        rm = torch.where(x255 <= 0, CUSTO_MOLHADO, um)
        y["unif"] = (x255 + simular_embutimento(x255, rp, rm, bits, rng)).clamp(0, 255)
        for k in sc:
            sc[k].append(disc(torch.cat([x255, y[k]])).float().cpu())
        rot.append(torch.cat([torch.zeros(len(x)), torch.ones(len(x))]))
    r = torch.cat(rot).numpy()
    return {f"auc_alvo_{k}": pe_auc(torch.cat(v).numpy(), r)[1] for k, v in sc.items()}


@torch.no_grad()
def validar(gerador, disc, loader, cfg, hill_ref, critico_w=None):
    dev = cfg["dispositivo"]
    gerador.eval(); disc.eval()
    if critico_w is not None:
        critico_w.eval()
    rng = torch.Generator().manual_seed(1234)
    rng_h = torch.Generator().manual_seed(999)
    bits, taxa, mse, scores, rotulos, n = 0.0, 0.0, 0.0, [], [], 0
    sc_hill, sc_w = [], []
    for x in loader:
        x255 = torch.round(x.to(dev) * 255.0)
        p = mapa_p(gerador, x255)
        bits += entropia_ternaria(p).sum().item()
        y = embutir(gerador, x255, cfg["bpp"], rng).float()
        d = y - x255
        taxa += (d != 0).float().mean(dim=(1, 2, 3)).sum().item()
        mse += d.pow(2).mean(dim=(1, 2, 3)).sum().item()
        scores.append(disc(torch.cat([x255, y])).float().cpu())
        sc_hill.append(disc(torch.cat([x255, estego_hill(x255, cfg["bpp"], rng_h).float()])).float().cpu())
        if critico_w is not None:
            sc_w.append(critico_w(torch.cat([x255, y])).float().cpu())
        rotulos.append(torch.cat([torch.zeros(len(x)), torch.ones(len(x))]))
        n += len(x)
    alvo = avaliar_laco_alvo(gerador, disc, loader, cfg) if cfg.get("bpp_laco_alvo") else {}
    gerador.train(); disc.train()
    if critico_w is not None:
        critico_w.train()
    amostras = cfg.get("canais", 1) * cfg["tamanho_img"] ** 2
    r = torch.cat(rotulos).numpy()
    pe, auc = pe_auc(torch.cat(scores).numpy(), r)
    extra = {"auc_laco_hill": pe_auc(torch.cat(sc_hill).numpy(), r)[1]}
    if sc_w:
        extra["auc_wgan"] = pe_auc(torch.cat(sc_w).numpy(), r)[1]
    return {**alvo, **extra, "bpp_mapa": bits / n / amostras, "taxa": taxa / n,
            "psnr": 10 * math.log10(255.0 ** 2 / max(mse / n, 1e-12)),
            "auc_laco": auc, "pe_laco": pe,
            "taxa_hill": hill_ref["taxa"], "psnr_hill": hill_ref["psnr"]}


@torch.no_grad()
def referencia_hill(loader, cfg):
    rng = torch.Generator().manual_seed(1234)
    taxa, mse, n = 0.0, 0.0, 0
    for x in loader:
        x255 = torch.round(x.to(cfg["dispositivo"]) * 255.0)
        rp, rm = custo_hill(x255)
        d = simular_embutimento(x255, rp, rm, cfg["bpp"] * x255[0].numel(), rng).float()
        d = (x255 + d).clamp(0, 255) - x255
        taxa += (d != 0).float().mean(dim=(1, 2, 3)).sum().item()
        mse += d.pow(2).mean(dim=(1, 2, 3)).sum().item()
        n += len(x)
    return {"taxa": taxa / n, "psnr": 10 * math.log10(255.0 ** 2 / max(mse / n, 1e-12))}


@torch.no_grad()
def salvar_amostras(gerador, loader, cfg, epoca):
    x = next(iter(loader))[:4].to(cfg["dispositivo"])
    x255 = torch.round(x * 255.0)
    p = mapa_p(gerador, x255)
    rp, _ = custo_hill(x255)
    inv_hill = (1.0 / rp).float()
    norm = lambda t: t / t.flatten(1).amax(1).clamp_min(1e-12).view(-1, 1, 1, 1)
    save_image(torch.cat([x, norm(p), norm(inv_hill)]), os.path.join(
        cfg["pasta_saida"], f"amostras_epoca{epoca:03d}.png"), nrow=4)




CONFIG = {
    "pasta_dados": None,
    "tamanho_img": 128,
    "canais": 1,
    "fracao_val": 0.05,
    "semente": 42,
    "max_imagens": None,

    "bpp": 0.4,
    "largura": 64,
    "blocos": 6,

    "discriminador": "srnet",
    "adversario": "wgan_gp",
    "lambda_wgan": 0.1,
    "critico_wgan_inicial": None,
    "n_d_wgan": 1,
    "alvo_adv": "literatura",
    "lambda_adv": 1.0,
    "lambda_cap": 100.0,
    "lambda_tanh": 60.0,
    "lambda_gp": 10.0,
    "lambda_z": 0.0,
    "lambda_drift": 0.0,
    "critico_inicial": None,
    "congelar_critico": (),
    "n_d": 5,
    "norma_critico": "afim",
    "bpp_laco_alvo": 0.0625,

    "lr_g": 1e-4,
    "lr_d": 1e-4,
    "tamanho_lote": 16,
    "epocas": 55,
    "limite_horas": None,
    "amostras_cada": 5,

    "auc_alvo": 0.55,
    "epocas_vencendo": 5,
    "parar_ao_vencer": False,
    "gerador_inicial": None,
    "base": "hill",

    "num_trabalhadores": 2,
    "pasta_saida": "runs/phase30",
    "espelho": None,
    "p_hill_float64": False,
    "dispositivo": "cuda" if torch.cuda.is_available() else "cpu",
}


COLUNAS = ["epoca", "perda_d", "perda_adv", "perda_cap", "bpp_treino", "bpp_mapa", "taxa",
           "taxa_hill", "psnr", "psnr_hill", "auc_laco", "pe_laco", "auc_alvo_braco",
           "auc_alvo_hill", "auc_alvo_unif", "auc_laco_hill", "perda_w", "adv_w", "auc_wgan",
           "vencendo", "segundos"]


def _colunas_do_csv(caminho):
    if not os.path.exists(caminho):
        return COLUNAS
    with open(caminho, encoding="utf-8") as f:
        return next(csv.reader(f), COLUNAS)


def venceu_epoca(val, cfg):
    return (val["auc_laco"] <= cfg["auc_alvo"]
            and abs(val["bpp_mapa"] - cfg["bpp"]) <= 0.1 * cfg["bpp"])


def _espelhar(cfg):
    if cfg.get("espelho"):
        f25.espelhar(cfg["pasta_saida"], cfg["espelho"])


def _achar_ckpt(cfg):
    for pasta in (cfg["pasta_saida"], cfg.get("espelho")):
        if pasta and os.path.exists(os.path.join(pasta, "ckpt_last.pt")):
            return os.path.join(pasta, "ckpt_last.pt")
    return None


def montar_loaders(cfg):
    cfg4 = dict(CONFIG_SPLIT, pasta_dados=cfg["pasta_dados"], semente_split=cfg["semente"],
                fracao_val=cfg["fracao_val"], max_imagens=cfg["max_imagens"])
    caminhos, idx_tr, idx_va, idx_te = dividir(cfg4)
    if not caminhos:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {cfg['pasta_dados']}")
    treino = [caminhos[i] for i in sorted(list(idx_tr) + list(idx_va))]
    val = [caminhos[i] for i in idx_te]
    print(f"[dados] {len(treino)} treino / {len(val)} val  ({cfg['pasta_dados']})")
    canais, tam = cfg.get("canais", 1), cfg["tamanho_img"]
    kw = dict(batch_size=cfg["tamanho_lote"], num_workers=cfg["num_trabalhadores"], pin_memory=True)
    return (DataLoader(base.DatasetImagens(treino, canais, tam, treino=True), shuffle=True,
                       drop_last=True, **kw),
            DataLoader(base.DatasetImagens(val, canais, tam, treino=False), shuffle=False, **kw))


def treinar(cfg=None):
    cfg = dict(CONFIG, **(cfg or {}))
    with p_hill_em(torch.float64 if cfg.get("p_hill_float64") else P_HILL_DTYPE):
        return _treinar(cfg)


def _treinar(cfg):
    dev = cfg["dispositivo"]
    os.makedirs(cfg["pasta_saida"], exist_ok=True)
    esp = cfg.get("espelho")
    if esp and os.path.isdir(esp) and os.path.abspath(esp) != os.path.abspath(cfg["pasta_saida"]):
        for nome in os.listdir(esp):
            origem, destino = os.path.join(esp, nome), os.path.join(cfg["pasta_saida"], nome)
            if os.path.isfile(origem) and not os.path.exists(destino):
                shutil.copy2(origem, destino)
    arquivo_log = open(os.path.join(cfg["pasta_saida"], "treino.log"), "a", encoding="utf-8")

    def log(msg=""):
        print(msg); arquivo_log.write(msg + "\n"); arquivo_log.flush()

    torch.manual_seed(cfg["semente"])
    loader_treino, loader_val = montar_loaders(cfg)

    p0 = p_uniforme(cfg["bpp"])
    gerador = GeradorCusto(cfg["canais"], cfg["largura"], cfg["blocos"], p0, cfg["base"],
                           cfg["bpp"]).to(dev)
    hibrido = cfg["adversario"] == "hibrido"
    if hibrido:
        disc = construir_discriminador(dict(cfg, adversario="bce", norma_critico="batch")).to(dev)
        critico_w = construir_discriminador(dict(cfg, adversario="wgan_gp", norma_critico="afim")).to(dev)
        if cfg.get("critico_wgan_inicial"):
            critico_w.load_state_dict(torch.load(cfg["critico_wgan_inicial"], map_location=dev,
                                                 weights_only=False))
        opt_w = torch.optim.Adam(critico_w.parameters(), lr=cfg["lr_d"], betas=(0.0, 0.9))
    else:
        disc = construir_discriminador(cfg).to(dev)
        critico_w = opt_w = None
    if cfg.get("critico_inicial"):
        disc.load_state_dict(torch.load(cfg["critico_inicial"], map_location=dev, weights_only=False))
    congelados = tuple(f"rede.{b}." for b in cfg.get("congelar_critico") or ())
    for nome, prm in disc.named_parameters():
        if congelados and nome.startswith(congelados):
            prm.requires_grad_(False)
    wgan = cfg["adversario"] == "wgan_gp"
    betas = (0.0, 0.9) if wgan else (0.5, 0.999)
    opt_g = torch.optim.Adam(gerador.parameters(), lr=cfg["lr_g"], betas=betas)
    opt_d = torch.optim.Adam([prm for prm in disc.parameters() if prm.requires_grad], lr=cfg["lr_d"],
                             betas=betas)

    inicio_ep = 1
    caminho = _achar_ckpt(cfg)
    if caminho:
        ck = torch.load(caminho, map_location="cpu", weights_only=False)
        gerador.load_state_dict(ck["gerador"]); disc.load_state_dict(ck["disc"])
        opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"])
        if hibrido:
            critico_w.load_state_dict(ck["critico_w"]); opt_w.load_state_dict(ck["opt_w"])
        inicio_ep = ck["epoca"] + 1
        log(f"[retomada] {caminho}: continuando da época {inicio_ep}")
    else:
        log("=" * 78)
        log(f"FASE 3.0 - custo aprendido  |  {cfg['adversario']} × {cfg['discriminador']}  |  "
            f"{cfg['bpp']} bpp  |  " + ("p inicial = HILL" if cfg["base"] == "hill" else f"p inicial {p0:.4f}"))
        log("config: " + ", ".join(f"{k}={v}" for k, v in sorted(cfg.items())))
        log("=" * 78)
        if cfg["gerador_inicial"]:
            ck = torch.load(cfg["gerador_inicial"], map_location="cpu", weights_only=False)
            gerador.load_state_dict(ck["gerador"])
            log(f"[inicial] gerador de {cfg['gerador_inicial']} (época {ck['epoca']}); "
                "adversário e otimizadores partem do zero")

    hill_ref = referencia_hill(loader_val, cfg)
    log(f"[hill  ] mesma carga, mesmas imagens de val: taxa {hill_ref['taxa']:.2%}  "
        f"PSNR {hill_ref['psnr']:.2f} dB")

    csv_metricas = os.path.join(cfg["pasta_saida"], "metricas.csv")
    amostras = cfg["canais"] * cfg["tamanho_img"] ** 2
    alvo = cfg["bpp"] * amostras
    colunas_csv = _colunas_do_csv(csv_metricas)
    t_inicio = time.time()
    epocas_feitas = 0
    vencendo = 0
    for epoca in range(inicio_ep, cfg["epocas"] + 1):
        t0 = time.time()
        soma = {"perda_d": 0.0, "perda_adv": 0.0, "perda_cap": 0.0, "bpp_treino": 0.0,
                "perda_w": 0.0, "adv_w": 0.0}
        passos = 0
        for x in loader_treino:
            x255 = torch.round(x.to(dev) * 255.0)
            b = len(x255)
            p = molhar(gerador(x255 / 255.0), x255)
            m = double_tanh(p, torch.rand_like(p), cfg["lambda_tanh"])
            y255 = (x255 + m).clamp(0, 255)

            for _ in range(cfg["n_d"]):
                s = disc(torch.cat([x255, y255.detach()]))
                if wgan:
                    perda_d = s[:b].mean() - s[b:].mean() + cfg["lambda_gp"] * \
                        fase2.penalidade_gradiente(disc, x255, y255.detach(), dev) + \
                        cfg.get("lambda_drift", 0.0) * s.pow(2).mean()
                else:
                    rot = torch.cat([torch.zeros(b, device=dev), torch.ones(b, device=dev)])
                    perda_d = F.binary_cross_entropy_with_logits(s, rot)
                opt_d.zero_grad(set_to_none=True); perda_d.backward(); opt_d.step()
                if hibrido:
                    break

            if hibrido:
                for _ in range(cfg["n_d_wgan"]):
                    sw = critico_w(torch.cat([x255, y255.detach()]))
                    perda_w = sw[:b].mean() - sw[b:].mean() + cfg["lambda_gp"] * \
                        fase2.penalidade_gradiente(critico_w, x255, y255.detach(), dev) + \
                        cfg.get("lambda_drift", 0.0) * sw.pow(2).mean()
                    opt_w.zero_grad(set_to_none=True); perda_w.backward(); opt_w.step()

            s = disc(torch.cat([x255, y255]))
            if wgan:
                perda_adv = s[b:].mean() - s[:b].mean()
            elif cfg["alvo_adv"] == "equilibrio":
                perda_adv = F.binary_cross_entropy_with_logits(s[b:], torch.full((b,), 0.5, device=dev))
            else:
                rot = torch.cat([torch.zeros(b, device=dev), torch.ones(b, device=dev)])
                perda_adv = -F.binary_cross_entropy_with_logits(s, rot)
            cap = entropia_ternaria(p)
            perda_cap = ((cap - alvo) / alvo).pow(2).mean()
            perda_g = cfg["lambda_adv"] * perda_adv + cfg["lambda_cap"] * perda_cap
            if hibrido:
                for prm in critico_w.parameters():
                    prm.requires_grad_(False)
                sw = critico_w(torch.cat([x255, y255]))
                for prm in critico_w.parameters():
                    prm.requires_grad_(True)
                adv_w = sw[b:].mean() - sw[:b].mean()
                perda_g = perda_g + cfg["lambda_wgan"] * adv_w
                soma["perda_w"] += perda_w.item(); soma["adv_w"] += adv_w.item()
            if cfg.get("lambda_z"):
                perda_g = perda_g + cfg["lambda_z"] * gerador.ultimo_z.pow(2).mean()
            opt_g.zero_grad(set_to_none=True); perda_g.backward(); opt_g.step()

            if not torch.isfinite(perda_g):
                raise FloatingPointError(f"perda do gerador não finita na época {epoca}")
            soma["perda_d"] += perda_d.item(); soma["perda_adv"] += perda_adv.item()
            soma["perda_cap"] += perda_cap.item()
            soma["bpp_treino"] += cap.mean().item() / amostras
            passos += 1

        val = validar(gerador, disc, loader_val, cfg, hill_ref, critico_w)
        vencendo = vencendo + 1 if venceu_epoca(val, cfg) else 0
        linha = {"epoca": epoca, **{k: v / max(passos, 1) for k, v in soma.items()}, **val,
                 "vencendo": vencendo, "segundos": time.time() - t0}
        estado = {"epoca": epoca, "cfg": cfg, "gerador": gerador.state_dict(),
                  "disc": disc.state_dict(), "opt_g": opt_g.state_dict(),
                  "opt_d": opt_d.state_dict(), "val": val}
        if hibrido:
            estado.update(critico_w=critico_w.state_dict(), opt_w=opt_w.state_dict())
        torch.save(estado, os.path.join(cfg["pasta_saida"], "ckpt_last.pt"))
        novo = not os.path.exists(csv_metricas)
        with open(csv_metricas, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=colunas_csv, extrasaction="ignore")
            if novo:
                w.writeheader()
            w.writerow({k: linha[k] for k in colunas_csv if k in linha})

        if cfg["amostras_cada"] and (epoca % cfg["amostras_cada"] == 0 or epoca == 1):
            salvar_amostras(gerador, loader_val, cfg, epoca)

        epocas_feitas += 1
        decorrido = time.time() - t_inicio
        restam = (cfg["epocas"] - epoca) * decorrido / epocas_feitas
        aviso = ""
        if abs(val["bpp_mapa"] - cfg["bpp"]) > 0.1 * cfg["bpp"]:
            aviso += "  <- CARGA FORA DO ALVO (>10%)"
        log(f"[época {epoca:3d}] D={linha['perda_d']:.4f}  adv={linha['perda_adv']:.4f}  "
            f"cap={linha['perda_cap']:.5f}  bpp={linha['bpp_treino']:.4f}  |  val: "
            f"bpp {val['bpp_mapa']:.4f}  taxa {val['taxa']:.2%} (HILL {val['taxa_hill']:.2%})  "
            f"PSNR {val['psnr']:.2f} (HILL {val['psnr_hill']:.2f})  AUC laço {val['auc_laco']:.4f} "
            f"(HILL {val['auc_laco_hill']:.4f})  "
            + (f"W={linha['perda_w']:.3f} advW={linha['adv_w']:.3f} AUC W {val['auc_wgan']:.4f}  "
               if hibrido else "")
            + (f"|  a {cfg['bpp_laco_alvo']:g} bpp: braço {val['auc_alvo_braco']:.4f}  HILL "
               f"{val['auc_alvo_hill']:.4f}  ±1 {val['auc_alvo_unif']:.4f}  "
               if cfg.get("bpp_laco_alvo") else "")
            + f"| {linha['segundos']:.0f}s, restam ~{restam / 60:.0f} min{aviso}")
        venceu = vencendo >= cfg["epocas_vencendo"]
        if vencendo == cfg["epocas_vencendo"]:
            torch.save(estado, os.path.join(cfg["pasta_saida"], "ckpt_venceu.pt"))
            log(f"[etapa ] adversário do laço vencido há {vencendo} épocas seguidas (AUC ≤ "
                f"{cfg['auc_alvo']}) -> ckpt_venceu.pt. Falta confirmar com o adversário fresco.")
        _espelhar(cfg)
        if venceu and cfg["parar_ao_vencer"]:
            log(f"[etapa ] parar_ao_vencer=True: parando na época {epoca}.")
            break

        if cfg["limite_horas"] and decorrido / 3600 + decorrido / epocas_feitas / 3600 > cfg["limite_horas"]:
            log(f"[limite] {cfg['limite_horas']} h atingido na época {epoca}; parando com o "
                "ckpt salvo. Rodar de novo continua daqui.")
            break

    arquivo_log.close()
    _espelhar(cfg)
    return os.path.join(cfg["pasta_saida"], "ckpt_last.pt")

