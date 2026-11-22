import torch
import torch.nn.functional as F

import fase0_codificador_decodificador as base
import fase25_residual as f25
import fase30_custo_wgan as f30


SIGMA_SUNIWARD = 1.0
PAD_SUNIWARD = 16
_HPDF = [-0.0544158422, 0.3128715909, -0.6756307363, 0.5853546837, 0.0158291053, -0.2840155430,
         -0.0004724846, 0.1287474266, 0.0173693010, -0.0440882539, -0.0139810279, 0.0087460940,
         0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768]


def _filtros_suniward(dtype, dev):
    h = torch.tensor(_HPDF, dtype=dtype, device=dev)
    sinal = torch.tensor([(-1.0) ** k for k in range(len(_HPDF))], dtype=dtype, device=dev)
    l = sinal * h.flip(0)
    return [torch.outer(l, h), torch.outer(h, l), torch.outer(h, h)]


def _conv_same_matlab(x, k):
    n = k.shape[-1]
    y = F.conv2d(x, k.flip(-1, -2).view(1, 1, n, n), padding=n - 1)
    s = (n - 1 + 1) // 2
    return y[..., s:s + x.shape[-2], s:s + x.shape[-1]]


def custo_suniward(x255):
    forma = x255.shape
    x = x255.reshape(-1, 1, *forma[-2:]).double()
    xp = f30._pad_simetrico(x, PAD_SUNIWARD)
    rho = torch.zeros_like(xp)
    for f in _filtros_suniward(x.dtype, x.device):
        r = _conv_same_matlab(xp, f)
        xi = 1.0 / (r.abs() + SIGMA_SUNIWARD)
        xi = torch.roll(xi, shifts=(1, 1), dims=(-2, -1))
        rho = rho + _conv_same_matlab(xi, f.abs().flip(-1, -2))
    p = PAD_SUNIWARD
    rho = rho[..., p:-p, p:-p]
    rho = torch.nan_to_num(rho, nan=f30.CUSTO_MOLHADO).clamp_max(f30.CUSTO_MOLHADO).reshape(forma)
    return (torch.where(x255 >= 255, f30.CUSTO_MOLHADO, rho),
            torch.where(x255 <= 0, f30.CUSTO_MOLHADO, rho))


def estego_suniward(x255, bpp, rng):
    rp, rm = custo_suniward(x255)
    return (x255 + f30.simular_embutimento(x255, rp, rm, bpp * x255[0].numel(), rng)).clamp(0, 255)


def estego_lsbr(x255, bpp, rng):
    n, amostras = x255.shape[0], x255[0].numel()
    usados = int(round(min(bpp, 1.0) * amostras))
    x = x255.reshape(n, -1).round().to(torch.int64).cpu()
    for i in range(n):
        pos = torch.randperm(amostras, generator=rng)[:usados]
        bit = torch.randint(0, 2, (usados,), generator=rng)
        x[i, pos] = (x[i, pos] & ~1) | bit
    return x.view(x255.shape).to(x255.device, x255.dtype)


_PROFUNDOS = {}


def _carregar_profundo(metodo, canais, dev):
    if metodo["ckpt"] in _PROFUNDOS:
        return _PROFUNDOS[metodo["ckpt"]]
    estado = torch.load(metodo["ckpt"], map_location=dev, weights_only=False)
    sd_c = estado.get("codificador", estado.get("encoder"))
    sd_d = estado.get("decodificador", estado.get("decoder"))
    c = estado.get("cfg") or {}
    f25.usar(metodo.get("arquitetura", c.get("arquitetura", "residual")))
    prof = int(c.get("profundidade_dados", 1))
    erro = None
    for larg in [c.get("largura"), 64, 32, 128]:
        if not larg:
            continue
        try:
            cod = base.Codificador(canais, prof, larg).to(dev)
            dec = base.Decodificador(canais, prof, larg).to(dev)
            cod.load_state_dict(sd_c)
            dec.load_state_dict(sd_d)
            _PROFUNDOS[metodo["ckpt"]] = (cod.eval(), dec.eval(), prof, larg)
            return _PROFUNDOS[metodo["ckpt"]]
        except RuntimeError as e:
            erro = e
    raise RuntimeError(f"{metodo['ckpt']}: pesos não cabem em nenhuma largura testada") from erro


@torch.no_grad()
def estego_profundo(metodo, x255, bpp, rng, estat):
    dev = x255.device
    canais, tam = x255.shape[1], x255.shape[-1]
    cod, dec, prof, _ = _carregar_profundo(metodo, canais, dev)
    tm = int(round(tam * bpp ** 0.5))
    assert abs((tm / tam) ** 2 - bpp) < 1e-9, f"carga {bpp} não é (tm/{tam})² para tm inteiro"
    cfg = {"tamanho_img": tam, "tamanho_msg": tm, "profundidade_dados": prof, "dispositivo": dev}
    msg = torch.randint(0, 2, (x255.shape[0], prof, tm, tm), generator=rng).float().to(dev)
    y = (cod(x255.float() / 255.0, base.espalhar_mensagem(msg, cfg)) * 255.0).round().clamp(0, 255)
    logits = base.agregar_logits(dec(y / 255.0), cfg)
    estat["acertos"] = estat.get("acertos", 0) + ((logits > 0).float() == msg).float().sum().item()
    estat["bits"] = estat.get("bits", 0) + msg.numel()
    return y.double()


def estego(metodo, x255, bpp, rng, estat):
    t = metodo["tipo"]
    if t == "suniward":
        return estego_suniward(x255, bpp, rng)
    if t == "lsbr":
        return estego_lsbr(x255, bpp, rng)
    if t == "profundo":
        return estego_profundo(metodo, x255, bpp, rng, estat)
    raise ValueError(f"tipo de método desconhecido: {t}")
