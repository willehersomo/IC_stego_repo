open('fase0_codificador_decodificador.py', 'w', encoding='utf-8').write(r'''import os
import glob
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image


CONFIG = {
  
    "pasta_dados": None,
    "canais": 3,
    "tamanho_img": 128,        
    "fracao_val": 0.05,       
    "max_imagens": None,      
    

    "profundidade_dados": 1,  
    "tamanho_msg": None,      


    "tamanho_lote": 16,       
    "epocas": 30,
    "taxa_aprend": 1e-3,      
    "largura": 32,             
    "lambda_imagem": 1.0,      
    "lambda_mensagem": 1.0,    
    "lambda_residuo": 0.0,    
    "usar_amp": True,          
    "alvo_bitacc": 0.99,       

    "num_trabalhadores": 2,   
    "semente": 42,             
    "pasta_saida": "runs/phase0",
    "salvar_cada": 1,        
    "dispositivo": "cuda" if torch.cuda.is_available() else "cpu",
}



COLAB_GPU = dict(
    tamanho_img=128,
    tamanho_msg=32,         
    tamanho_lote=32,
    epocas=120,
    largura=64,
    max_imagens=None,
    num_trabalhadores=4,
    lambda_imagem=2.0,
    lambda_residuo=0.05,
)


EXTENSOES_IMG = (".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".ppm", ".tif", ".tiff", ".webp")


class DatasetImagens(Dataset):
    def __init__(self, caminhos, canais, tamanho_img, treino=True):
        self.caminhos = caminhos
        self.canais = canais
        self.modo = "L" if canais == 1 else "RGB"
        if treino:
            self.transformacao = transforms.Compose([
                transforms.Resize(tamanho_img),
                transforms.RandomCrop(tamanho_img, pad_if_needed=True),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transformacao = transforms.Compose([
                transforms.Resize(tamanho_img),
                transforms.CenterCrop(tamanho_img),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.caminhos)

    def __getitem__(self, indice):
        imagem = Image.open(self.caminhos[indice]).convert(self.modo)
        return self.transformacao(imagem)


class DatasetSintetico(Dataset):
    def __init__(self, canais, tamanho_img, n=512):
        self.canais, self.tamanho_img, self.n = canais, tamanho_img, n

    def __len__(self):
        return self.n

    def __getitem__(self, indice):
        base = torch.rand(self.canais, 1, 1)
        gradiente = torch.linspace(0, 1, self.tamanho_img).view(1, 1, -1)
        return (base + 0.3 * gradiente).clamp(0, 1).expand(
            self.canais, self.tamanho_img, self.tamanho_img).clone()


def montar_loaders(cfg):
    if cfg["pasta_dados"]:
        caminhos = []
        for ext in EXTENSOES_IMG:
            caminhos += glob.glob(os.path.join(cfg["pasta_dados"], "**", "*" + ext), recursive=True)
            caminhos += glob.glob(os.path.join(cfg["pasta_dados"], "**", "*" + ext.upper()), recursive=True)
        caminhos = sorted(set(caminhos))
        if not caminhos:
            raise FileNotFoundError(f"Nenhuma imagem encontrada em {cfg['pasta_dados']}")
        random.Random(cfg["semente"]).shuffle(caminhos)
        if cfg.get("max_imagens"):
            caminhos = caminhos[:cfg["max_imagens"]]
            print(f"[dados] max_imagens={cfg['max_imagens']} -> usando {len(caminhos)} imagens")
        n_val = max(1, int(len(caminhos) * cfg["fracao_val"]))
        caminhos_val, caminhos_treino = caminhos[:n_val], caminhos[n_val:]
        print(f"[dados] {len(caminhos_treino)} treino / {len(caminhos_val)} val  ({cfg['pasta_dados']})")
        ds_treino = DatasetImagens(caminhos_treino, cfg["canais"], cfg["tamanho_img"], treino=True)
        ds_val = DatasetImagens(caminhos_val, cfg["canais"], cfg["tamanho_img"], treino=False)
    else:
        print("[dados] pasta_dados=None -> usando dataset SINTÉTICO (teste rápido)")
        ds_treino = DatasetSintetico(cfg["canais"], cfg["tamanho_img"], n=512)
        ds_val = DatasetSintetico(cfg["canais"], cfg["tamanho_img"], n=64)

    loader_treino = DataLoader(ds_treino, batch_size=cfg["tamanho_lote"], shuffle=True,
                               num_workers=cfg["num_trabalhadores"], drop_last=True, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=cfg["tamanho_lote"], shuffle=False,
                            num_workers=cfg["num_trabalhadores"], pin_memory=True)
    return loader_treino, loader_val


def bloco_conv(canais_entrada, canais_saida):
    return nn.Sequential(
        nn.Conv2d(canais_entrada, canais_saida, 3, padding=1),
        nn.BatchNorm2d(canais_saida),
        nn.LeakyReLU(0.2, inplace=True),
    )


class Codificador(nn.Module):

    def __init__(self, canais, profundidade_dados, largura=32):
        super().__init__()
        self.bloco1 = bloco_conv(canais, largura)
        self.bloco2 = bloco_conv(largura + profundidade_dados, largura)
        self.bloco3 = bloco_conv(2 * largura + profundidade_dados, largura)
        self.para_residuo = nn.Conv2d(3 * largura + profundidade_dados, canais, 3, padding=1)

    def forward(self, portadora, mensagem):
        a = self.bloco1(portadora)
        b = self.bloco2(torch.cat([a, mensagem], dim=1))
        c = self.bloco3(torch.cat([a, b, mensagem], dim=1))
        residuo = self.para_residuo(torch.cat([a, b, c, mensagem], dim=1))
        estego = (portadora + residuo).clamp(0.0, 1.0)
        return estego


class Decodificador(nn.Module):
    

    def __init__(self, canais, profundidade_dados, largura=32):
        super().__init__()
        self.bloco1 = bloco_conv(canais, largura)
        self.bloco2 = bloco_conv(largura, largura)
        self.bloco3 = bloco_conv(2 * largura, largura)
        self.para_logits = nn.Conv2d(3 * largura, profundidade_dados, 3, padding=1)

    def forward(self, estego):
        a = self.bloco1(estego)
        b = self.bloco2(a)
        c = self.bloco3(torch.cat([a, b], dim=1))
        return self.para_logits(torch.cat([a, b, c], dim=1))


def _tamanho_msg(cfg):
    return cfg.get("tamanho_msg") or cfg["tamanho_img"]


def mensagem_aleatoria(tamanho_lote, cfg):
    tm = _tamanho_msg(cfg)
    return torch.randint(0, 2, (tamanho_lote, cfg["profundidade_dados"], tm, tm),
                         dtype=torch.float32, device=cfg["dispositivo"])


def espalhar_mensagem(mensagem, cfg):
    if _tamanho_msg(cfg) == cfg["tamanho_img"]:
        return mensagem
    return F.interpolate(mensagem, size=cfg["tamanho_img"], mode="nearest")


def agregar_logits(logits, cfg):
    if _tamanho_msg(cfg) == cfg["tamanho_img"]:
        return logits
    return F.adaptive_avg_pool2d(logits, _tamanho_msg(cfg))


@torch.no_grad()
def acuracia_bits(logits, mensagem):
    predito = (torch.sigmoid(logits) > 0.5).float()
    return (predito == mensagem).float().mean().item()


def _janela_gaussiana(tamanho, sigma, canais, dispositivo):
    coords = torch.arange(tamanho, device=dispositivo) - tamanho // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    janela2d = (g[:, None] * g[None, :])
    return janela2d.expand(canais, 1, tamanho, tamanho).contiguous()


@torch.no_grad()
def ssim(a, b, canais, tamanho_janela=11, sigma=1.5):

    janela = _janela_gaussiana(tamanho_janela, sigma, canais, a.device)
    pad = tamanho_janela // 2
    mu_a = F.conv2d(a, janela, padding=pad, groups=canais)
    mu_b = F.conv2d(b, janela, padding=pad, groups=canais)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    var_a = F.conv2d(a * a, janela, padding=pad, groups=canais) - mu_a2
    var_b = F.conv2d(b * b, janela, padding=pad, groups=canais) - mu_b2
    cov_ab = F.conv2d(a * b, janela, padding=pad, groups=canais) - mu_ab
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_ab + c1) * (2 * cov_ab + c2)) / ((mu_a2 + mu_b2 + c1) * (var_a + var_b + c2))
    return s.mean().item()

def treinar():
    cfg = CONFIG
    torch.manual_seed(cfg["semente"]); random.seed(cfg["semente"])
    os.makedirs(cfg["pasta_saida"], exist_ok=True)
    dispositivo = cfg["dispositivo"]
    tm = _tamanho_msg(cfg)
    bpp = cfg["profundidade_dados"] * (tm / cfg["tamanho_img"]) ** 2
    print(f"[infra] dispositivo={dispositivo}  payload={bpp:.4g} bpp  "
          f"(tamanho_msg={tm}, tamanho_img={cfg['tamanho_img']})")

    loader_treino, loader_val = montar_loaders(cfg)

    codificador = Codificador(cfg["canais"], cfg["profundidade_dados"], cfg["largura"]).to(dispositivo)
    decodificador = Decodificador(cfg["canais"], cfg["profundidade_dados"], cfg["largura"]).to(dispositivo)


    otimizador = torch.optim.Adam(
        list(codificador.parameters()) + list(decodificador.parameters()), lr=cfg["taxa_aprend"])
    usar_amp = cfg["usar_amp"] and dispositivo == "cuda"
    escalador = torch.cuda.amp.GradScaler(enabled=usar_amp)

    melhor = {"score": float("-inf"), "epoca": 0, "acuracia_bits": 0.0, "psnr": 0.0, "ssim": 0.0}

    for epoca in range(1, cfg["epocas"] + 1):
        codificador.train(); decodificador.train()
        acumulado = {"img": 0.0, "msg": 0.0, "acc": 0.0, "n": 0}

        for portadora in loader_treino:
            portadora = portadora.to(dispositivo, non_blocking=True)
            mensagem = mensagem_aleatoria(portadora.size(0), cfg)
            mensagem_espalhada = espalhar_mensagem(mensagem, cfg)

            with torch.cuda.amp.autocast(enabled=usar_amp):
                estego = codificador(portadora, mensagem_espalhada)
                logits = agregar_logits(decodificador(estego), cfg)

                perda_imagem = F.mse_loss(estego, portadora)
                perda_mensagem = F.binary_cross_entropy_with_logits(logits, mensagem)
                perda_residuo = (estego - portadora).abs().mean()
                perda = (cfg["lambda_imagem"] * perda_imagem
                         + cfg["lambda_mensagem"] * perda_mensagem
                         + cfg["lambda_residuo"] * perda_residuo)


            otimizador.zero_grad()
            escalador.scale(perda).backward()
            escalador.step(otimizador)
            escalador.update()

            n_lote = portadora.size(0)
            acumulado["img"] += perda_imagem.item() * n_lote
            acumulado["msg"] += perda_mensagem.item() * n_lote
            acumulado["acc"] += acuracia_bits(logits, mensagem) * n_lote
            acumulado["n"] += n_lote

        n = acumulado["n"]
        print(f"[treino] época {epoca:3d}  img_mse={acumulado['img']/n:.5f}  "
              f"msg_bce={acumulado['msg']/n:.4f}  acuracia_bits={acumulado['acc']/n:.4f}")

       
        val = avaliar(codificador, decodificador, loader_val, cfg)
        ok = "OK" if (val["acuracia_bits"] >= cfg["alvo_bitacc"] and val["psnr"] >= 35.0) else "  "
        print(f"[  val ] época {epoca:3d}  acuracia_bits={val['acuracia_bits']:.4f}  "
              f"psnr={val['psnr']:.2f}dB  ssim={val['ssim']:.4f}  linf={val['linf']:.4f}  {ok}")

      
        score = (val["psnr"] if val["acuracia_bits"] >= cfg["alvo_bitacc"]
                 else val["acuracia_bits"] - 100.0)
        if score > melhor["score"]:
            melhor = {"score": score, "epoca": epoca,
                      **{k: val[k] for k in ("acuracia_bits", "psnr", "ssim")}}
            torch.save({"codificador": codificador.state_dict(),
                        "decodificador": decodificador.state_dict(),
                        "epoca": epoca, "cfg": cfg, "val": val},
                       os.path.join(cfg["pasta_saida"], "ckpt_best.pt"))

        if epoca % cfg["salvar_cada"] == 0:
            salvar_amostras(codificador, loader_val, cfg, epoca)
            torch.save({"codificador": codificador.state_dict(),
                        "decodificador": decodificador.state_dict(),
                        "epoca": epoca, "cfg": cfg},
                       os.path.join(cfg["pasta_saida"], "ckpt_last.pt"))

    print(f"\n[fim] checkpoints e amostras em: {cfg['pasta_saida']}")
    print(f"[melhor] época {melhor['epoca']}: acuracia_bits={melhor['acuracia_bits']:.4f}  "
          f"psnr={melhor['psnr']:.2f}dB  ssim={melhor['ssim']:.4f}")
    if melhor["acuracia_bits"] >= cfg["alvo_bitacc"] and melhor["psnr"] >= 35.0:
        print("[Fase 0 FECHADA] acurácia e PSNR>=35dB atingidos -> pode ligar a WGAN-GP (Fase 2).")
    else:
        print("[Fase 0 ainda aberta] baixar o payload (tamanho_msg menor) e/ou treinar mais antes da Fase 2.")
    return {"acuracia_bits": melhor["acuracia_bits"], "psnr": melhor["psnr"], "ssim": melhor["ssim"]}


@torch.no_grad()
def avaliar(codificador, decodificador, loader, cfg):
    codificador.eval(); decodificador.eval()
    acc, mse, ss, linf, n = 0.0, 0.0, 0.0, 0.0, 0
    for portadora in loader:
        portadora = portadora.to(cfg["dispositivo"])
        mensagem = mensagem_aleatoria(portadora.size(0), cfg)
        estego = codificador(portadora, espalhar_mensagem(mensagem, cfg))
        logits = agregar_logits(decodificador(estego), cfg)
        n_lote = portadora.size(0)
        acc += acuracia_bits(logits, mensagem) * n_lote
        mse += F.mse_loss(estego, portadora).item() * n_lote
        ss += ssim(estego, portadora, cfg["canais"]) * n_lote
        linf = max(linf, (estego - portadora).abs().max().item())
        n += n_lote
    mse = mse / max(n, 1)
    psnr = 10.0 * math.log10(1.0 / mse) if mse > 0 else float("inf")
    return {"acuracia_bits": acc / max(n, 1), "psnr": psnr, "ssim": ss / max(n, 1), "linf": linf}


@torch.no_grad()
def salvar_amostras(codificador, loader, cfg, epoca):
    codificador.eval()
    portadora = next(iter(loader)).to(cfg["dispositivo"])[:8]
    mensagem = mensagem_aleatoria(portadora.size(0), cfg)
    estego = codificador(portadora, espalhar_mensagem(mensagem, cfg))
    residuo = (estego - portadora).abs()
    residuo = residuo / (residuo.amax() + 1e-8)
    grade = torch.cat([portadora, estego, residuo], dim=0)
    caminho = os.path.join(cfg["pasta_saida"], f"samples_epoch{epoca:03d}.png")
    save_image(grade, caminho, nrow=8)


if __name__ == "__main__":
    treinar()
''')

open('fase2_wgan_gp.py', 'w', encoding='utf-8').write(r'''import os
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import fase0_codificador_decodificador as base


CONFIG = dict(base.CONFIG)
CONFIG.update({
    "lambda_adv": 0.01,       
                               
    "n_critic": 5,             
    "lambda_gp": 10.0,         
    "largura_critico": 64,     
    "lr_critico": 1e-4,       
    "beta1": 0.0, "beta2": 0.9,

    "taxa_aprend": 1e-4,      
    "epocas": 60,
    "usar_amp": False,
    "carregar_fase0": "runs/phase0_bossbase/ckpt_best.pt",
    "pasta_saida": "runs/phase2_bossbase",
})

COLAB_GPU = dict(base.COLAB_GPU)
COLAB_GPU.update({
    "epocas": 60,
    "lambda_adv": 0.01,
    "usar_amp": False,
})


class Critico(nn.Module):

    def __init__(self, canais, largura=64):
        super().__init__()

        def bloco(entrada, saida, norm=True):
            camadas = [nn.Conv2d(entrada, saida, 4, stride=2, padding=1)]
            if norm:
                camadas.append(nn.InstanceNorm2d(saida, affine=True))
            camadas.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*camadas)

        self.corpo = nn.Sequential(
            bloco(canais, largura, norm=False),
            bloco(largura, largura * 2),
            bloco(largura * 2, largura * 4),
            bloco(largura * 4, largura * 8),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.para_score = nn.Linear(largura * 8, 1)

    def forward(self, x):
        h = self.corpo(x)
        h = self.pool(h).flatten(1)
        return self.para_score(h).squeeze(1)


def penalidade_gradiente(critico, reais, falsas, dispositivo):

    b = reais.size(0)
    eps = torch.rand(b, 1, 1, 1, device=dispositivo)
    interp = (eps * reais + (1.0 - eps) * falsas).requires_grad_(True)
    score = critico(interp)
    grads = torch.autograd.grad(
        outputs=score, inputs=interp,
        grad_outputs=torch.ones_like(score),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    grads = grads.reshape(b, -1)
    return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()


def auc_binaria(scores, rotulos):
    scores = scores.detach().float().flatten()
    rotulos = rotulos.detach().float().flatten()
    n_pos = int(rotulos.sum().item())
    n_neg = int(rotulos.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ordem = torch.argsort(scores)
    postos = torch.empty_like(scores)
    postos[ordem] = torch.arange(1, scores.numel() + 1, dtype=scores.dtype, device=scores.device)
    soma_postos_pos = postos[rotulos == 1].sum().item()
    return (soma_postos_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


@torch.no_grad()
def avaliar_critico(codificador, critico, loader, cfg):
    codificador.eval(); critico.eval()
    scores, rotulos = [], []
    for portadora in loader:
        portadora = portadora.to(cfg["dispositivo"])
        b = portadora.size(0)
        mensagem = base.mensagem_aleatoria(b, cfg)
        estego = codificador(portadora, base.espalhar_mensagem(mensagem, cfg))
        scores.append(critico(torch.cat([portadora, estego], dim=0)).cpu())
        rotulos.append(torch.cat([torch.zeros(b), torch.ones(b)]))
    codificador.train(); critico.train()
    return auc_binaria(torch.cat(scores), torch.cat(rotulos))


def treinar():
    cfg = CONFIG
    torch.manual_seed(cfg["semente"]); random.seed(cfg["semente"])
    os.makedirs(cfg["pasta_saida"], exist_ok=True)
    dispositivo = cfg["dispositivo"]
    tm = base._tamanho_msg(cfg)
    bpp = cfg["profundidade_dados"] * (tm / cfg["tamanho_img"]) ** 2
    print(f"[infra] dispositivo={dispositivo}  payload={bpp:.4g} bpp  "
          f"lambda_adv={cfg['lambda_adv']}  n_critic={cfg['n_critic']}")

    loader_treino, loader_val = base.montar_loaders(cfg)

    codificador = base.Codificador(cfg["canais"], cfg["profundidade_dados"], cfg["largura"]).to(dispositivo)
    decodificador = base.Decodificador(cfg["canais"], cfg["profundidade_dados"], cfg["largura"]).to(dispositivo)
    critico = Critico(cfg["canais"], cfg["largura_critico"]).to(dispositivo)

    ck = cfg.get("carregar_fase0")
    if ck and os.path.exists(ck):
        estado = torch.load(ck, map_location=dispositivo)
        codificador.load_state_dict(estado["codificador"])
        decodificador.load_state_dict(estado["decodificador"])
        print(f"[warm start] pesos da Fase 0 carregados de {ck}")
    else:
        print(f"[warm start] {ck!r} não encontrado -> treinando do zero "
              "(recomendado rodar a Fase 0 antes)")

    opt_g = torch.optim.Adam(list(codificador.parameters()) + list(decodificador.parameters()),
                             lr=cfg["taxa_aprend"], betas=(cfg["beta1"], cfg["beta2"]))
    opt_c = torch.optim.Adam(critico.parameters(),
                             lr=cfg["lr_critico"], betas=(cfg["beta1"], cfg["beta2"]))

    melhor = {"score": float("-inf"), "epoca": 0, "acuracia_bits": 0.0, "psnr": 0.0, "ssim": 0.0}

    for epoca in range(1, cfg["epocas"] + 1):
        codificador.train(); decodificador.train(); critico.train()
        ac = {"c": 0.0, "w": 0.0, "adv": 0.0, "img": 0.0, "msg": 0.0, "acc": 0.0, "n": 0}

        for portadora in loader_treino:
            portadora = portadora.to(dispositivo, non_blocking=True)
            b = portadora.size(0)

            for _ in range(cfg["n_critic"]):
                mensagem = base.mensagem_aleatoria(b, cfg)
                with torch.no_grad():
                    estego = codificador(portadora, base.espalhar_mensagem(mensagem, cfg))
                score_real = critico(portadora).mean()
                score_falsa = critico(estego).mean()
                gp = penalidade_gradiente(critico, portadora, estego, dispositivo)
                perda_c = score_falsa - score_real + cfg["lambda_gp"] * gp
                opt_c.zero_grad(); perda_c.backward(); opt_c.step()

            w_dist = (score_real - score_falsa).item()

            mensagem = base.mensagem_aleatoria(b, cfg)
            estego = codificador(portadora, base.espalhar_mensagem(mensagem, cfg))
            logits = base.agregar_logits(decodificador(estego), cfg)

            perda_img = F.mse_loss(estego, portadora)
            perda_msg = F.binary_cross_entropy_with_logits(logits, mensagem)
            perda_res = (estego - portadora).abs().mean()
            perda_adv = -critico(estego).mean()
            perda_g = (cfg["lambda_imagem"] * perda_img
                       + cfg["lambda_mensagem"] * perda_msg
                       + cfg["lambda_residuo"] * perda_res
                       + cfg["lambda_adv"] * perda_adv)
            opt_g.zero_grad(); perda_g.backward(); opt_g.step()

            ac["c"] += perda_c.item() * b;   ac["w"] += w_dist * b
            ac["adv"] += perda_adv.item() * b
            ac["img"] += perda_img.item() * b; ac["msg"] += perda_msg.item() * b
            ac["acc"] += base.acuracia_bits(logits, mensagem) * b
            ac["n"] += b

        n = ac["n"]
        print(f"[treino] época {epoca:3d}  perda_c={ac['c']/n:+.3f}  W={ac['w']/n:+.3f}  "
              f"adv={ac['adv']/n:+.3f}  img_mse={ac['img']/n:.5f}  "
              f"msg_bce={ac['msg']/n:.4f}  acuracia_bits={ac['acc']/n:.4f}")

        val = base.avaliar(codificador, decodificador, loader_val, cfg)
        auc_c = avaliar_critico(codificador, critico, loader_val, cfg)
        ok = "OK" if (val["acuracia_bits"] >= cfg["alvo_bitacc"] and val["psnr"] >= 35.0) else "  "
        print(f"[  val ] época {epoca:3d}  acuracia_bits={val['acuracia_bits']:.4f}  "
              f"psnr={val['psnr']:.2f}dB  ssim={val['ssim']:.4f}  linf={val['linf']:.4f}  "
              f"auc_critico={auc_c:.4f} (efetiva {max(auc_c, 1 - auc_c):.4f})  {ok}")

        score = (val["psnr"] if val["acuracia_bits"] >= cfg["alvo_bitacc"]
                 else val["acuracia_bits"] - 100.0)
        if score > melhor["score"]:
            melhor = {"score": score, "epoca": epoca,
                      **{k: val[k] for k in ("acuracia_bits", "psnr", "ssim")}}
            torch.save({"codificador": codificador.state_dict(),
                        "decodificador": decodificador.state_dict(),
                        "critico": critico.state_dict(),
                        "epoca": epoca, "cfg": cfg, "val": val},
                       os.path.join(cfg["pasta_saida"], "ckpt_best.pt"))

        if epoca % cfg["salvar_cada"] == 0:
            base.salvar_amostras(codificador, loader_val, cfg, epoca)
            torch.save({"codificador": codificador.state_dict(),
                        "decodificador": decodificador.state_dict(),
                        "critico": critico.state_dict(),
                        "epoca": epoca, "cfg": cfg},
                       os.path.join(cfg["pasta_saida"], "ckpt_last.pt"))

    print(f"\n[fim] checkpoints e amostras em: {cfg['pasta_saida']}")
    print(f"[melhor] época {melhor['epoca']}: acuracia_bits={melhor['acuracia_bits']:.4f}  "
          f"psnr={melhor['psnr']:.2f}dB  ssim={melhor['ssim']:.4f}")
    return {"acuracia_bits": melhor["acuracia_bits"], "psnr": melhor["psnr"], "ssim": melhor["ssim"]}


if __name__ == "__main__":
    treinar()
''')

open('fase25_residual.py', 'w', encoding='utf-8').write(r'''import os
import shutil
import sys
import time

import torch
import torch.nn as nn

import fase0_codificador_decodificador as base
import fase2_wgan_gp as fase2


class BlocoResidual(nn.Module):

    def __init__(self, canais):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(canais, canais, 3, padding=1),
            nn.BatchNorm2d(canais),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(canais, canais, 3, padding=1),
            nn.BatchNorm2d(canais),
        )

    def forward(self, x):
        return x + self.conv(x)


class CodificadorResidual(nn.Module):
  

    def __init__(self, canais, profundidade_dados, largura=32, blocos=3):
        super().__init__()
        self.features = base.bloco_conv(canais, largura)
        self.entrada = base.bloco_conv(largura + profundidade_dados, largura)
        self.residuais = nn.Sequential(*[BlocoResidual(largura) for _ in range(blocos)])
        self.para_residuo = nn.Conv2d(largura, canais, 3, padding=1)

    def forward(self, portadora, mensagem):
        x = self.features(portadora)
        x = self.entrada(torch.cat([x, mensagem], dim=1))
        x = self.residuais(x)
        residuo = self.para_residuo(x)
        return (portadora + residuo).clamp(0.0, 1.0)


class DecodificadorResidual(nn.Module):

    def __init__(self, canais, profundidade_dados, largura=32, blocos=3):
        super().__init__()
        self.entrada = base.bloco_conv(canais, largura)
        self.residuais = nn.Sequential(*[BlocoResidual(largura) for _ in range(blocos)])
        self.para_logits = nn.Conv2d(largura, profundidade_dados, 3, padding=1)

    def forward(self, estego):
        x = self.entrada(estego)
        x = self.residuais(x)
        return self.para_logits(x)


_ORIGINAIS = {"Codificador": base.Codificador, "Decodificador": base.Decodificador}
_ATUAL = "denso"

_EXTRAS = {}


def registrar(nome, fabrica_cod, fabrica_dec=None):
    _EXTRAS[nome] = (fabrica_cod, fabrica_dec)
    return nome

BLOCOS_RESIDUAIS = 3


def usar(arquitetura):

    global _ATUAL
    if arquitetura in _EXTRAS:
        fab_cod, fab_dec = _EXTRAS[arquitetura]
        base.Codificador = fab_cod
        base.Decodificador = fab_dec or (
            lambda c, p, l=32: DecodificadorResidual(c, p, l, BLOCOS_RESIDUAIS))
        _ATUAL = arquitetura
        print(f"[arquitetura] gerador = {arquitetura}")
        return arquitetura

    if arquitetura not in ("denso", "residual"):
        raise ValueError('arquitetura deve ser "denso", "residual" ou uma registrada: '
                         + ", ".join(sorted(_EXTRAS)) or "(nenhuma)")

    if arquitetura == "denso":
        base.Codificador = _ORIGINAIS["Codificador"]
        base.Decodificador = _ORIGINAIS["Decodificador"]
    else:
        def _cod(canais, profundidade, largura=32):
            return CodificadorResidual(canais, profundidade, largura, BLOCOS_RESIDUAIS)

        def _dec(canais, profundidade, largura=32):
            return DecodificadorResidual(canais, profundidade, largura, BLOCOS_RESIDUAIS)

        base.Codificador = _cod
        base.Decodificador = _dec

    _ATUAL = arquitetura
    print(f"[arquitetura] gerador = {arquitetura}")
    return arquitetura


def contar_parametros(canais=1, profundidade=1, largura=64):
 
    resultado = {}
    for nome, (cod, dec) in {
        "denso": (_ORIGINAIS["Codificador"], _ORIGINAIS["Decodificador"]),
        "residual": (
            lambda c, p, l: CodificadorResidual(c, p, l, BLOCOS_RESIDUAIS),
            lambda c, p, l: DecodificadorResidual(c, p, l, BLOCOS_RESIDUAIS),
        ),
    }.items():
        c = cod(canais, profundidade, largura)
        d = dec(canais, profundidade, largura)
        resultado[nome] = {
            "codificador": sum(p.numel() for p in c.parameters()),
            "decodificador": sum(p.numel() for p in d.parameters()),
        }
        resultado[nome]["total"] = (resultado[nome]["codificador"]
                                    + resultado[nome]["decodificador"])
    return resultado


class _Tee:

    def __init__(self, caminho):
        self.caminho = caminho
        self.arquivo = None
        self.stdout = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.caminho) or ".", exist_ok=True)
        self.arquivo = open(self.caminho, "a", encoding="utf-8")
        self.stdout = sys.stdout
        sys.stdout = self
        return self

    def write(self, texto):
        self.stdout.write(texto)
        self.arquivo.write(texto)
        self.arquivo.flush()

    def flush(self):
        self.stdout.flush()
        self.arquivo.flush()

    def __exit__(self, *exc):
        sys.stdout = self.stdout
        self.arquivo.close()


def _gerar_csvs(log, pasta):
    try:
        from logs_para_csv import ler_log, gravar
    except ImportError:
        print("[csv] logs_para_csv.py não encontrado; CSVs não gerados")
        return
    treino, val = ler_log(log)
    for linhas, nome in ((treino, "metricas_treino.csv"), (val, "metricas_val.csv")):
        if linhas:
            gravar(linhas, os.path.join(pasta, nome))
            print(f"[csv] {nome}: {len(linhas)} linhas")


def espelhar(pasta, destino):
    if not destino or os.path.abspath(destino) == os.path.abspath(pasta):
        return
    os.makedirs(destino, exist_ok=True)
    copiados = 0
    for nome in os.listdir(pasta):
        origem = os.path.join(pasta, nome)
        if os.path.isfile(origem):
            shutil.copy2(origem, os.path.join(destino, nome))
            copiados += 1
    print(f"[espelho] {copiados} arquivo(s) copiado(s) para {destino}")


def _rodar(modulo, cfg_extra, pasta_saida, espelho, rotulo):
    rotulo = f"{rotulo} ({_ATUAL})"
    modulo.CONFIG.update(base.COLAB_GPU)
    modulo.CONFIG.update(cfg_extra)
    modulo.CONFIG["pasta_saida"] = pasta_saida
    os.makedirs(pasta_saida, exist_ok=True)
    log = os.path.join(pasta_saida, "treino.log")

    inicio = time.time()
    with _Tee(log):
        print(f"\n===== {rotulo} iniciado em "
              f"{time.strftime('%Y-%m-%d %H:%M:%S')} =====")
        print(f"[arquitetura] {_ATUAL}  |  blocos residuais = {BLOCOS_RESIDUAIS}")
        print(f"[cfg] " + "  ".join(
            f"{k}={modulo.CONFIG[k]}" for k in
            ("tamanho_img", "tamanho_msg", "tamanho_lote", "largura", "epocas",
             "lambda_imagem", "lambda_residuo") if k in modulo.CONFIG))
        resultado = modulo.treinar()
        minutos = (time.time() - inicio) / 60
        print(f"[tempo ] {rotulo}: {minutos:.1f} min "
              f"({minutos * 60 / max(modulo.CONFIG['epocas'], 1):.1f} s/época)")

    _gerar_csvs(log, pasta_saida)
    espelhar(pasta_saida, espelho)
    return resultado


def treinar_fase0(pasta_saida, espelho=None, **cfg_extra):
    return _rodar(base, cfg_extra, pasta_saida, espelho, "Fase 0")


def treinar_fase2(pasta_saida, espelho=None, **cfg_extra):
    return _rodar(fase2, cfg_extra, pasta_saida, espelho, "Fase 2")


ALVO_ACURACIA = 0.99
ALVO_PSNR_FASE2 = 36.30


def _ler_csv(caminho):
    import csv
    if not os.path.exists(caminho):
        return []
    with open(caminho, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def comparar(csv_denso, csv_residual, alvo_psnr=ALVO_PSNR_FASE2, rotulo="Fase 2"):
 
    d, r = _ler_csv(csv_denso), _ler_csv(csv_residual)
    if not r:
        print(f"[comparar] {csv_residual} não encontrado - rode o treino antes.")
        return None
    if not d:
        print(f"[comparar] {csv_denso} não encontrado - sem linha de base para comparar.")
        return None

    n = min(len(d), len(r))
    print(f"\n{'=' * 74}\n{rotulo}: DENSO × RESIDUAL nas primeiras {n} épocas\n{'=' * 74}")

    def resumo(linhas, nome):
        linhas = linhas[:n]
        psnr = [float(x["psnr"]) for x in linhas if x.get("psnr")]
        acc = [float(x["acuracia_bits"]) for x in linhas if x.get("acuracia_bits")]
        ssim = [float(x["ssim"]) for x in linhas if x.get("ssim")]
        melhor = max(psnr) if psnr else 0.0
        ult10 = psnr[-10:] or [0.0]
        print(f"  {nome:<9} melhor PSNR {melhor:6.2f} dB | "
              f"média 10 últimas {sum(ult10) / len(ult10):6.2f} dB | "
              f"acurácia mín {min(acc) if acc else 0:.4f} | "
              f"SSIM máx {max(ssim) if ssim else 0:.4f} | "
              f"épocas >= {alvo_psnr:.2f} dB: {sum(1 for p in psnr if p >= alvo_psnr)}")
        return {"melhor": melhor, "acc_min": min(acc) if acc else 0.0,
                "media10": sum(ult10) / len(ult10)}

    rd, rr = resumo(d, "denso"), resumo(r, "residual")

    print(f"\n  diferença (residual - denso): "
          f"{rr['melhor'] - rd['melhor']:+.2f} dB no melhor, "
          f"{rr['media10'] - rd['media10']:+.2f} dB na média das 10 últimas")

    print(f"\n{'-' * 74}\nCRITÉRIOS DE SAÍDA DA FASE 2.5")
    c1 = rr["acc_min"] >= ALVO_ACURACIA
    c2 = rr["melhor"] >= alvo_psnr
    print(f"  (1) acurácia de bits >= {ALVO_ACURACIA}      -> {rr['acc_min']:.4f}  "
          f"{'OK' if c1 else 'FALHOU'}")
    print(f"  (2) PSNR >= {alvo_psnr:.2f} dB (não regredir) -> {rr['melhor']:.2f}  "
          f"{'OK' if c2 else 'FALHOU'}")
    print("  (3) custo por época medido            -> ver a linha [tempo ] em cada treino.log")
    print("  (4) os dois geradores preservados     -> confira o espelho no Drive")

    if c1 and c2:
        print("\n  Residual passa. Decisão de qual segue: maior PSNR a igual acurácia;")
        print("  empate técnico (< 0,2 dB) fica com o mais barato por época.")
    else:
        print("\n  Residual NÃO bateu os critérios. Resultado negativo também é resultado:")
        print("  registre no ROADMAP e siga com o denso.")
    return {"denso": rd, "residual": rr}
''')

open('fase3_esteganalisador.py', 'w', encoding='utf-8').write(r'''import os
import csv
import time
import random
import shutil
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

import fase0_codificador_decodificador as base
import fase2_wgan_gp as fase2
import fase25_residual as fase25


CONFIG = dict(fase2.CONFIG)
CONFIG.update({

    "arquitetura": "residual",

    "alvo_psnr": None,

    "lambda_srnet": 0.01,
    "n_srnet": 1,
    "alvo_srnet": "equilibrio",
    "lr_srnet": 1e-4,
    "srnet_compacta": False,

    "epocas": 60,
    "usar_amp": False,
    "carregar_fase0": None,
    "carregar_fase2": "runs/phase2_residual/ckpt_best.pt",


    "retomar_de": None,

    "salvar_otimizadores": True,
    "pasta_saida": "runs/phase3_bossbase",

    "espelho_ckpt": None,     
})

COLAB_GPU = dict(fase2.COLAB_GPU)
COLAB_GPU.update({
    "epocas": 60,
    "lambda_srnet": 0.01,
    "usar_amp": False,

    "num_trabalhadores": max(1, min(4, (os.cpu_count() or 2) - 1)),
})


class _Tipo1(nn.Module):
    def __init__(self, entrada, saida):
        super().__init__()
        self.conv = nn.Conv2d(entrada, saida, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(saida)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class _Tipo2(nn.Module):
    def __init__(self, canais):
        super().__init__()
        self.c1 = nn.Conv2d(canais, canais, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(canais)
        self.c2 = nn.Conv2d(canais, canais, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(canais)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)), inplace=True)
        h = self.b2(self.c2(h))
        return x + h


class _Tipo3(nn.Module):
    def __init__(self, entrada, saida):
        super().__init__()
        self.c1 = nn.Conv2d(entrada, saida, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(saida)
        self.c2 = nn.Conv2d(saida, saida, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(saida)
        self.pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.atalho_conv = nn.Conv2d(entrada, saida, 1, stride=2, bias=False)
        self.atalho_bn = nn.BatchNorm2d(saida)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)), inplace=True)
        h = self.pool(self.b2(self.c2(h)))
        return self.atalho_bn(self.atalho_conv(x)) + h


class _Tipo4(nn.Module):
    def __init__(self, entrada, saida):
        super().__init__()
        self.c = nn.Conv2d(entrada, saida, 3, padding=1, bias=False)
        self.b = nn.BatchNorm2d(saida)

    def forward(self, x):
        h = F.relu(self.b(self.c(x)), inplace=True)
        return h.mean(dim=(2, 3))


class SRNet(nn.Module):
    def __init__(self, canais, compacta=False):
        super().__init__()
        e = 0.5 if compacta else 1.0
        f = lambda n: max(4, int(n * e))

        self.tipo1 = nn.Sequential(_Tipo1(canais, f(64)), _Tipo1(f(64), f(16)))
        self.tipo2 = nn.Sequential(*[_Tipo2(f(16)) for _ in range(5)])
        self.tipo3 = nn.Sequential(
            _Tipo3(f(16), f(16)),
            _Tipo3(f(16), f(64)),
            _Tipo3(f(64), f(128)),
            _Tipo3(f(128), f(256)),
        )
        self.tipo4 = _Tipo4(f(256), f(512))
        self.para_logit = nn.Linear(f(512), 1)

    def forward(self, x):
        h = self.tipo1(x)
        h = self.tipo2(h)
        h = self.tipo3(h)
        h = self.tipo4(h)
        return self.para_logit(h).squeeze(1)

auc_binaria = fase2.auc_binaria


@torch.no_grad()
def avaliar_srnet(codificador, srnet, loader, cfg):

    dispositivo = cfg["dispositivo"]
    codificador.eval(); srnet.eval()
    todos_scores, todos_rotulos = [], []

    for portadora in loader:
        portadora = portadora.to(dispositivo, non_blocking=True)
        b = portadora.size(0)
        mensagem = base.mensagem_aleatoria(b, cfg)
        estego = codificador(portadora, base.espalhar_mensagem(mensagem, cfg))

        logits = srnet(torch.cat([portadora, estego], dim=0)).cpu()
        todos_scores.append(logits)
        todos_rotulos.append(torch.cat([torch.zeros(b), torch.ones(b)]))

    scores = torch.cat(todos_scores)
    rotulos = torch.cat(todos_rotulos)
    auc = auc_binaria(scores, rotulos)

    predito = (scores > 0).float()
    vp = float(((predito == 1) & (rotulos == 1)).sum())
    fp = float(((predito == 1) & (rotulos == 0)).sum())
    fn = float(((predito == 0) & (rotulos == 1)).sum())
    acc = (predito == rotulos).float().mean().item()
    precisao = vp / (vp + fp) if (vp + fp) > 0 else float("nan")
    recall = vp / (vp + fn) if (vp + fn) > 0 else float("nan")
    f1 = (2 * precisao * recall / (precisao + recall)
          if (precisao == precisao and recall == recall and precisao + recall > 0)
          else float("nan"))
    return {"auc": auc, "acuracia_deteccao": acc, "precisao": precisao,
            "recall": recall, "f1": f1}


def _carregar_gerador(caminho, codificador, decodificador, critico, dispositivo, log=print):

    if not caminho or not os.path.exists(caminho):
        log(f"[warm start] {caminho!r} não encontrado -> gerador DO ZERO.\n"
              "             A Fase 3 é para ser construída SOBRE a Fase 2. Rode a Fase 2 antes,\n"
              "             ou aponte 'carregar_fase2' para o ckpt_best.pt no Drive.")
        return None

    estado = torch.load(caminho, map_location=dispositivo)
    cod = estado.get("codificador", estado.get("encoder"))
    dec = estado.get("decodificador", estado.get("decoder"))
    if cod is None or dec is None:
        raise KeyError(f"{caminho}: não achei os pesos do codificador/decodificador. "
                       f"Chaves presentes: {list(estado.keys())}")
    try:
        codificador.load_state_dict(cod)
        decodificador.load_state_dict(dec)
    except RuntimeError as e:
        raise RuntimeError(f"{caminho}: os pesos não cabem no gerador '{fase25._ATUAL}'. "
                           "Confira cfg['arquitetura'] - checkpoint denso só carrega em gerador "
                           "denso, residual só em residual.") from e

    if "critico" in estado:
        critico.load_state_dict(estado["critico"])
        log(f"[warm start] gerador + crítico carregados de {caminho} (época {estado.get('epoca', '?')})")
    else:
        log(f"[warm start] gerador carregado de {caminho}; crítico DO ZERO "
              "(checkpoint não tinha os pesos do crítico)")
    return estado.get("val", {}).get("psnr")


class _Log:
 
    def __init__(self, caminho):
        self.caminho = caminho
        self.arquivo = open(caminho, "a", encoding="utf-8")

    def __call__(self, texto=""):
        print(texto)
        self.arquivo.write(texto + "\n")
        self.arquivo.flush()

    def fechar(self):
        self.arquivo.close()


def _espelhar(cfg, origem, log=print):
    destino = cfg.get("espelho_ckpt")
    if not destino:
        return
  
    if os.path.abspath(destino) == os.path.abspath(cfg["pasta_saida"]):
        return
    try:
        os.makedirs(destino, exist_ok=True)
        shutil.copy2(origem, os.path.join(destino, os.path.basename(origem)))
    except Exception as e:
        log(f"[espelho] falhou ao copiar para {destino}: {e}")


def treinar():
    cfg = CONFIG
    torch.manual_seed(cfg["semente"]); random.seed(cfg["semente"])
    os.makedirs(cfg["pasta_saida"], exist_ok=True)
    dispositivo = cfg["dispositivo"]
    tm = base._tamanho_msg(cfg)
    bpp = cfg["profundidade_dados"] * (tm / cfg["tamanho_img"]) ** 2

    log = _Log(os.path.join(cfg["pasta_saida"], "treino.log"))
    log(f"===== Fase 3 iniciada em {datetime.now():%Y-%m-%d %H:%M:%S} =====")
    log("[cfg] " + "  ".join(f"{k}={cfg[k]}" for k in
        ("tamanho_img", "tamanho_msg", "tamanho_lote", "largura", "epocas",
         "lambda_imagem", "lambda_mensagem", "lambda_residuo", "lambda_adv",
         "lambda_srnet", "n_critic", "n_srnet", "lr_srnet", "srnet_compacta", "arquitetura")))
    log(f"[infra] dispositivo={dispositivo}  payload={bpp:.4g} bpp  "
          f"lambda_adv={cfg['lambda_adv']}  lambda_srnet={cfg['lambda_srnet']}  "
          f"n_critic={cfg['n_critic']}  n_srnet={cfg['n_srnet']}"
          + ("  [SRNet COMPACTA]" if cfg["srnet_compacta"] else ""))

    loader_treino, loader_val = base.montar_loaders(cfg)

    fase25.usar(cfg["arquitetura"])
    codificador = base.Codificador(cfg["canais"], cfg["profundidade_dados"], cfg["largura"]).to(dispositivo)
    decodificador = base.Decodificador(cfg["canais"], cfg["profundidade_dados"], cfg["largura"]).to(dispositivo)
    critico = fase2.Critico(cfg["canais"], cfg["largura_critico"]).to(dispositivo)
    srnet = SRNet(cfg["canais"], compacta=cfg["srnet_compacta"]).to(dispositivo)

    psnr_partida = _carregar_gerador(cfg.get("carregar_fase2"), codificador, decodificador,
                                     critico, dispositivo, log=log)
    if cfg["alvo_psnr"] is not None:
        alvo_psnr = cfg["alvo_psnr"]
        log(f"[alvo  ] critério (2): PSNR >= {alvo_psnr:.2f} dB (fixado em cfg)")
    elif psnr_partida is not None:
        alvo_psnr = round(psnr_partida - 1.0, 2)
        log(f"[alvo  ] critério (2): PSNR >= {alvo_psnr:.2f} dB (partida {psnr_partida:.2f} - 1 dB)")
    else:
        alvo_psnr = 37.27
        log(f"[alvo  ] checkpoint sem PSNR gravado -> critério (2) no padrão da Fase 2.5: "
            f"PSNR >= {alvo_psnr:.2f} dB")
    n_par = sum(p.numel() for p in srnet.parameters())
    log(f"[srnet] iniciada DO ZERO - {n_par/1e6:.2f}M parâmetros")

    opt_g = torch.optim.Adam(list(codificador.parameters()) + list(decodificador.parameters()),
                             lr=cfg["taxa_aprend"], betas=(cfg["beta1"], cfg["beta2"]))
    opt_c = torch.optim.Adam(critico.parameters(),
                             lr=cfg["lr_critico"], betas=(cfg["beta1"], cfg["beta2"]))
    opt_s = torch.optim.Adam(srnet.parameters(), lr=cfg["lr_srnet"])

    melhor = {"score": float("-inf"), "epoca": 0, "acuracia_bits": 0.0, "psnr": 0.0,
              "ssim": 0.0, "auc": float("nan")}
    epoca_inicial = 1
    ck_ret = cfg.get("retomar_de")
    if ck_ret and os.path.exists(ck_ret):
        est = torch.load(ck_ret, map_location=dispositivo)
        codificador.load_state_dict(est["codificador"])
        decodificador.load_state_dict(est["decodificador"])
        critico.load_state_dict(est["critico"])
        srnet.load_state_dict(est["srnet"])
        if "opt_g" in est:
            opt_g.load_state_dict(est["opt_g"])
            opt_c.load_state_dict(est["opt_c"])
            opt_s.load_state_dict(est["opt_s"])
        else:
            log("[retomada] o checkpoint não tem o estado dos otimizadores: o Adam recomeça "
                "do zero e as primeiras épocas dão um solavanco.")
        if est.get("melhor"):
            melhor = est["melhor"]
        if est.get("rng") is not None:
            torch.set_rng_state(est["rng"].cpu())
        epoca_inicial = int(est["epoca"]) + 1
        log(f"[retomada] {ck_ret} - continuando da época {epoca_inicial} "
            f"(melhor até aqui: época {melhor['epoca']}, PSNR {melhor['psnr']:.2f} dB)")
        if epoca_inicial > cfg["epocas"]:
            log(f"[retomada] o run já tinha {est['epoca']} épocas e cfg['epocas'] = "
                f"{cfg['epocas']}. Suba 'epocas' para continuar.")
    elif ck_ret:
        log(f"[retomada] {ck_ret!r} não encontrado -> começando do zero.")

    retomando = epoca_inicial > 1
    caminho_csv = os.path.join(cfg["pasta_saida"], "metricas_val.csv")
    if not (retomando and os.path.exists(caminho_csv)):
        with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoca", "acuracia_bits", "psnr", "ssim", "auc_srnet",
                                    "auc_critico", "w_dist", "acuracia_det", "precisao",
                                    "recall", "f1"])

   
    caminho_csv_treino = os.path.join(cfg["pasta_saida"], "metricas_treino.csv")
    if not (retomando and os.path.exists(caminho_csv_treino)):
        with open(caminho_csv_treino, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoca", "perda_c", "w_dist", "adv", "srnet_bce", "enganar",
                                    "img_mse", "msg_bce", "acuracia_bits"])

    t_inicio = time.time()
    for epoca in range(epoca_inicial, cfg["epocas"] + 1):
        t_epoca = time.time()
        codificador.train(); decodificador.train(); critico.train(); srnet.train()
        ac = {"c": 0.0, "w": 0.0, "adv": 0.0, "s": 0.0, "eng": 0.0,
              "img": 0.0, "msg": 0.0, "acc": 0.0, "n": 0}

        for portadora in loader_treino:
            portadora = portadora.to(dispositivo, non_blocking=True)
            b = portadora.size(0)

            for _ in range(cfg["n_critic"]):
                mensagem = base.mensagem_aleatoria(b, cfg)
                with torch.no_grad():
                    estego = codificador(portadora, base.espalhar_mensagem(mensagem, cfg))
                score_real = critico(portadora).mean()
                score_falsa = critico(estego).mean()
                gp = fase2.penalidade_gradiente(critico, portadora, estego, dispositivo)
                perda_c = score_falsa - score_real + cfg["lambda_gp"] * gp
                opt_c.zero_grad(); perda_c.backward(); opt_c.step()

            w_dist = (score_real - score_falsa).item()

         
            for _ in range(cfg["n_srnet"]):
                mensagem = base.mensagem_aleatoria(b, cfg)
                with torch.no_grad():
                    estego = codificador(portadora, base.espalhar_mensagem(mensagem, cfg))
                logits_s = srnet(torch.cat([portadora, estego], dim=0))
                alvos_s = torch.cat([torch.zeros(b, device=dispositivo),
                                     torch.ones(b, device=dispositivo)])
                perda_s = F.binary_cross_entropy_with_logits(logits_s, alvos_s)
                opt_s.zero_grad(); perda_s.backward(); opt_s.step()

            mensagem = base.mensagem_aleatoria(b, cfg)
            estego = codificador(portadora, base.espalhar_mensagem(mensagem, cfg))
            logits = base.agregar_logits(decodificador(estego), cfg)

            perda_img = F.mse_loss(estego, portadora)
            perda_msg = F.binary_cross_entropy_with_logits(logits, mensagem)
            perda_res = (estego - portadora).abs().mean()
            perda_adv = -critico(estego).mean()

  
            logits_g = srnet(torch.cat([portadora.detach(), estego], dim=0))
            logit_est_g = logits_g[b:]
            valor_alvo = 0.5 if cfg["alvo_srnet"] == "equilibrio" else 0.0
            perda_eng = F.binary_cross_entropy_with_logits(
                logit_est_g, torch.full_like(logit_est_g, valor_alvo))

            perda_g = (cfg["lambda_imagem"] * perda_img
                       + cfg["lambda_mensagem"] * perda_msg
                       + cfg["lambda_residuo"] * perda_res
                       + cfg["lambda_adv"] * perda_adv
                       + cfg["lambda_srnet"] * perda_eng)
            opt_g.zero_grad(); perda_g.backward(); opt_g.step()

            ac["c"] += perda_c.item() * b;    ac["w"] += w_dist * b
            ac["adv"] += perda_adv.item() * b; ac["s"] += perda_s.item() * b
            ac["eng"] += perda_eng.item() * b
            ac["img"] += perda_img.item() * b; ac["msg"] += perda_msg.item() * b
            ac["acc"] += base.acuracia_bits(logits, mensagem) * b
            ac["n"] += b

        n = ac["n"]
        seg = time.time() - t_epoca
        feitas = epoca - epoca_inicial + 1
        restam = (cfg["epocas"] - epoca) * (time.time() - t_inicio) / feitas
        log(f"[tempo ] época {epoca:3d}  {seg:6.1f}s  ({ac['n']//cfg['tamanho_lote']} passos de gerador)"
            f"  restam ~{restam/60:.0f} min")
        log(f"[treino] época {epoca:3d}  perda_c={ac['c']/n:+.3f}  W={ac['w']/n:+.3f}  "
              f"srnet_bce={ac['s']/n:.4f}  enganar={ac['eng']/n:.4f}  "
              f"img_mse={ac['img']/n:.5f}  msg_bce={ac['msg']/n:.4f}  "
              f"acuracia_bits={ac['acc']/n:.4f}")

        val = base.avaliar(codificador, decodificador, loader_val, cfg)
        det = avaliar_srnet(codificador, srnet, loader_val, cfg)
        auc, auc_ef = det["auc"], max(det["auc"], 1.0 - det["auc"])
        auc_c = fase2.avaliar_critico(codificador, critico, loader_val, cfg)

        c1 = val["acuracia_bits"] >= cfg["alvo_bitacc"]
        c2 = val["psnr"] >= alvo_psnr
        c3 = abs(auc - 0.5) <= 0.05
        marca = "".join(["1" if c1 else ".", "2" if c2 else ".", "3" if c3 else "."])
        log(f"[  val ] época {epoca:3d}  acuracia_bits={val['acuracia_bits']:.4f}  "
              f"psnr={val['psnr']:.2f}dB  ssim={val['ssim']:.4f}  "
              f"auc_srnet={auc:.4f} (efetiva {auc_ef:.4f})  "
              f"auc_critico={auc_c:.4f} (efetiva {max(auc_c, 1 - auc_c):.4f})  "
              f"criterios[{marca}]")

 
        log(f"[detect] época {epoca:3d}  acuracia={det['acuracia_deteccao']:.4f}  "
            f"precisao={det['precisao']:.4f}  recall={det['recall']:.4f}  "
            f"f1={det['f1']:.4f}")

        with open(caminho_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoca, f"{val['acuracia_bits']:.4f}", f"{val['psnr']:.2f}",
                                    f"{val['ssim']:.4f}", f"{auc:.4f}", f"{auc_c:.4f}",
                                    f"{ac['w']/n:.4f}", f"{det['acuracia_deteccao']:.4f}",
                                    f"{det['precisao']:.4f}", f"{det['recall']:.4f}",
                                    f"{det['f1']:.4f}"])
        with open(caminho_csv_treino, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoca, f"{ac['c']/n:.4f}", f"{ac['w']/n:.4f}",
                                    f"{ac['adv']/n:.4f}", f"{ac['s']/n:.4f}", f"{ac['eng']/n:.4f}",
                                    f"{ac['img']/n:.6f}", f"{ac['msg']/n:.4f}", f"{ac['acc']/n:.4f}"])


        for reg in (caminho_csv, caminho_csv_treino, os.path.join(cfg["pasta_saida"], "treino.log")):
            _espelhar(cfg, reg, log)

      
        score = (val["psnr"] if c1 else val["acuracia_bits"] - 100.0)
        if score > melhor["score"]:
            melhor = {"score": score, "epoca": epoca, "auc": auc, "det": det,
                      **{k: val[k] for k in ("acuracia_bits", "psnr", "ssim")}}
            destino = os.path.join(cfg["pasta_saida"], "ckpt_best.pt")
            torch.save({"codificador": codificador.state_dict(),
                        "decodificador": decodificador.state_dict(),
                        "critico": critico.state_dict(),
                        "srnet": srnet.state_dict(),
                        "epoca": epoca, "cfg": cfg, "val": val, "det": det},
                       destino)
            _espelhar(cfg, destino, log)

        if epoca % cfg["salvar_cada"] == 0:
            base.salvar_amostras(codificador, loader_val, cfg, epoca)
            destino = os.path.join(cfg["pasta_saida"], "ckpt_last.pt")
            estado = {"codificador": codificador.state_dict(),
                      "decodificador": decodificador.state_dict(),
                      "critico": critico.state_dict(),
                      "srnet": srnet.state_dict(),
                      "epoca": epoca, "cfg": cfg,
                      "melhor": melhor, "rng": torch.get_rng_state()}
            if cfg.get("salvar_otimizadores", True):
                estado.update(opt_g=opt_g.state_dict(), opt_c=opt_c.state_dict(),
                              opt_s=opt_s.state_dict())
            torch.save(estado, destino)
            _espelhar(cfg, destino, log)

    auc_ef = max(melhor["auc"], 1.0 - melhor["auc"])
    log(f"\n[fim] checkpoints, amostras e metricas_val.csv em: {cfg['pasta_saida']}")
    log(f"[melhor] época {melhor['epoca']}: acuracia_bits={melhor['acuracia_bits']:.4f}  "
          f"psnr={melhor['psnr']:.2f}dB  ssim={melhor['ssim']:.4f}  auc_srnet={melhor['auc']:.4f}")
    log("\nCRITÉRIOS DE SAÍDA DA FASE 3:")
    log(f"  (1) acuracia_bits >= 0.99          -> {melhor['acuracia_bits']:.4f}  "
          f"{'OK' if melhor['acuracia_bits'] >= 0.99 else 'FALHOU: baixe lambda_srnet'}")
    log(f"  (2) PSNR >= {alvo_psnr:.2f} dB (partida - 1dB) -> {melhor['psnr']:.2f}  "
          f"{'OK' if melhor['psnr'] >= alvo_psnr else 'FALHOU: baixe lambda_srnet ou suba lambda_imagem'}")
    if melhor["acuracia_bits"] < cfg["alvo_bitacc"]:
        veredito = ("SEM VALOR: a mensagem não é recuperada - o gerador engana o detector "
                    "deixando de esconder. Baixe lambda_srnet")
    elif abs(melhor["auc"] - 0.5) <= 0.05:
        veredito = "OK"
    elif melhor["auc"] > 0.5:
        veredito = "detector separando: suba lambda_srnet"
    else:
        veredito = ("gerador PASSOU DO PONTO (estego com score menor que portadora). "
                    "NÃO suba lambda_srnet - isso piora. Use alvo_srnet='equilibrio'")
    log(f"  (3) AUC do laço ~ 0.5              -> {melhor['auc']:.4f} (efetiva {auc_ef:.4f})  {veredito}")
    d = melhor.get("det") or {}
    if d:
        log(f"      detector na melhor época: acuracia={d['acuracia_deteccao']:.4f}  "
            f"precisao={d['precisao']:.4f}  recall={d['recall']:.4f}  f1={d['f1']:.4f}")
        if d["recall"] > 0.95 and d["precisao"] < 0.55:
            log("      ATENÇÃO: recall ~1 com precisão ~0,5 = o detector responde 'estego' "
                "para tudo. Isso é detector que NAO CONVERGIU, nao seguranca.")
    log("\nATENÇÃO: o critério (3) mede o detector que participou do treino. Ele NÃO prova")
    log("segurança - pode significar que aquela SRNet não convergiu. A prova é a Fase 4,")
    log("que treina um esteganalisador independente e mede P_E. Ver ROADMAP.md.")
    log(f"\n[registros] treino.log, metricas_val.csv e metricas_treino.csv em {cfg['pasta_saida']}")
    log(f"[retomada] se a sessão cair, rode de novo com "
        f"cfg['retomar_de'] = '{os.path.join(cfg['pasta_saida'], 'ckpt_last.pt')}'")
    if not cfg.get("espelho_ckpt"):
        log("\nLEMBRETE: 'espelho_ckpt' está vazio. Se isto rodou no Colab, BAIXE AGORA o")
        log("ckpt_best.pt E os arquivos de registro - foi assim que os pesos das Fases 0 e 2")
        log("se perderam, e a saída da célula do notebook some junto com o runtime.")
    else:
        _espelhar(cfg, os.path.join(cfg["pasta_saida"], "treino.log"), log)
        log(f"[espelho] registros e checkpoints copiados para {cfg['espelho_ckpt']}")
    log.fechar()
    return {"acuracia_bits": melhor["acuracia_bits"], "psnr": melhor["psnr"],
            "ssim": melhor["ssim"], "auc": melhor["auc"], "alvo_psnr": alvo_psnr}


if __name__ == "__main__":
    treinar()
''')

open('fase30_custo_wgan.py', 'w', encoding='utf-8').write(r'''import contextlib
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

''')

open('fase40_pe.py', 'w', encoding='utf-8').write(r'''import csv
import math
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fase0_codificador_decodificador as base
import fase3_esteganalisador as f3
import fase30_custo_wgan as f30


CONFIG = {
    "pasta_dados": None,
    "conjunto": "boss",
    "tamanho_img": 128,
    "canais": 1,
    "semente_split": 42,
    "fracao_val": 0.05,
    "n_val_detector": 1000,
    "max_imagens": None,
    "semente_embutir": 7,
    "epocas": 20,
    "pares_por_lote": 32,
    "lr": {"srnet": 1e-3, "srnet_cur": 1e-3, "effnet": 5e-4, "effnet_cur": 5e-4},
    "lr_fino": 5e-4,
    "curriculo": {"srnet_cur": (1.0, 0.4, 0.2),
                  "effnet_cur": (1.0, 0.4, 0.2)},
    "epocas_curriculo": 5,
    "decaimento": 1e-4,
    "usar_amp": True,
    "razao_taxa_max": 1.05,
    "salvar_pesos": True,
    "limite_horas": None,
    "pasta_saida": "runs/phase40",
    "espelho": None,
    "dispositivo": "cuda" if torch.cuda.is_available() else "cpu",
}



pe_auc = f30.pe_auc


def quarteto(scores, rotulos):
    s, y = np.asarray(scores), np.asarray(rotulos)
    p = (s > 0).astype(np.int64)
    vp, fp, fn = ((p == 1) & (y == 1)).sum(), ((p == 1) & (y == 0)).sum(), ((p == 0) & (y == 1)).sum()
    prec = vp / (vp + fp) if vp + fp else float("nan")
    rec = vp / (vp + fn) if vp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else float("nan")
    return {"acuracia": float((p == y).mean()), "precisao": float(prec),
            "recall": float(rec), "f1": float(f1)}


@torch.no_grad()
def metricas_imagem(capas, estego, lote=100):
    x, y = capas.float(), estego.float()
    d = y - x
    ss = sum(base.ssim(y[i:i + lote] / 255.0, x[i:i + lote] / 255.0, x.shape[1]) * len(x[i:i + lote])
             for i in range(0, len(x), lote)) / len(x)
    mse = d.pow(2).mean().item()
    return {"psnr": float(10 * math.log10(255.0 ** 2 / max(mse, 1e-12))),
            "ssim": float(ss), "mse": float(mse),
            "taxa": float((d != 0).float().mean()),
            "rms": float(math.sqrt(mse))}



def dividir(cfg):
    c = dict(f30.CONFIG_SPLIT, pasta_dados=cfg["pasta_dados"], semente_split=cfg["semente_split"],
             fracao_val=cfg["fracao_val"], n_val_detector=cfg["n_val_detector"],
             max_imagens=cfg["max_imagens"])
    return f30.dividir(c)


@torch.no_grad()
def carregar_capas(caminhos, cfg, log=print):
    cache = os.path.join(cfg["pasta_saida"], "corpus",
                         f"capas_{cfg['canais']}x{cfg['tamanho_img']}_{len(caminhos)}.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    ds = base.DatasetImagens(caminhos, cfg["canais"], cfg["tamanho_img"], treino=False)
    t0 = time.time()
    capas = torch.stack([torch.round(ds[i] * 255.0).to(torch.uint8) for i in range(len(ds))])
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save(capas, cache)
    log(f"[corpus] {len(capas)} portadoras em {time.time() - t0:.0f}s")
    return capas


def _lsbm(capas, bpp, semente=7):
    g = torch.Generator().manual_seed(semente)
    n, amostras = capas.shape[0], capas[0].numel()
    usados = int(round(min(bpp, 1.0) * amostras))
    x = capas.reshape(n, -1).to(torch.int16).clone()
    for i in range(n):
        pos = torch.randperm(amostras, generator=g)[:usados]
        muda = torch.rand(usados, generator=g) < 0.5
        sinal = torch.where(torch.rand(usados, generator=g) < 0.5, -1, 1).to(torch.int16)
        v = x[i, pos]
        sinal = torch.where(v == 0, 1, torch.where(v == 255, -1, sinal)).to(torch.int16)
        x[i, pos] = v + sinal * muda.to(torch.int16)
    return x.clamp(0, 255).to(torch.uint8).view_as(capas)


@torch.no_grad()
def gerar_estego(metodo, capas, idx_te, bpp, cfg, log=print):
    cache = os.path.join(cfg["pasta_saida"], "corpus",
                         f"estego_{metodo['nome']}_{bpp:g}_{cfg['canais']}x{cfg['tamanho_img']}_{len(capas)}.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    dev, t0 = cfg["dispositivo"], time.time()
    rng = torch.Generator().manual_seed(cfg["semente_embutir"])
    lote = f30._lote_por_tamanho(64, capas)
    ger = f30.carregar_gerador(metodo["ckpt"], dev) if metodo["tipo"] == "custo" else None
    partes, estat = [], {}
    for i in range(0, len(capas), lote):
        x = capas[i:i + lote].to(dev).double()
        if metodo["tipo"] == "hill":
            y = f30.estego_hill(x, bpp, rng)
        elif metodo["tipo"] == "unif":
            y = f30.estego_unif(x, bpp, rng)
        elif metodo["tipo"] == "custo":
            with f30.p_hill_em(torch.float64):
                y = f30.embutir(ger, x.float(), bpp, rng)
        else:
            import fase46_baselines as f46
            y = f46.estego(metodo, x, bpp, rng, estat)
        partes.append(y.float().round().clamp(0, 255).to(torch.uint8).cpu())
    estego = torch.cat(partes)
    info = metricas_imagem(capas[idx_te], estego[idx_te])
    if estat.get("bits"):
        info["acuracia_bits"] = estat["acertos"] / estat["bits"]
        log(f"[corpus] {metodo['nome']}: acurácia de bits do decodificador após arredondar {info['acuracia_bits']:.4f}")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save((estego, info), cache)
    log(f"[corpus] {metodo['nome']} @ {bpp:g} bpp: {len(estego)} estegos em {time.time() - t0:.0f}s | "
        f"teste: PSNR {info['psnr']:.2f}  SSIM {info['ssim']:.5f}  MSE {info['mse']:.4f}  "
        f"taxa {info['taxa']:.2%}")
    return estego, info



class _EffNet(nn.Module):

    def __init__(self, canais=1, pretreinada=True):
        super().__init__()
        import timm
        try:
            self.rede = timm.create_model("efficientnet_b0", pretrained=pretreinada,
                                          in_chans=canais, num_classes=1)
            self.pretreinada = pretreinada
        except Exception as erro:
            print(f"[effnet] pesos do ImageNet indisponíveis ({type(erro).__name__}); do zero")
            self.rede = timm.create_model("efficientnet_b0", pretrained=False,
                                          in_chans=canais, num_classes=1)
            self.pretreinada = False
        self.rede.conv_stem.stride = (1, 1)

    def forward(self, x):
        return self.rede((x - 0.449) / 0.226).squeeze(1)


def construir_detector(nome, canais=1):
    if nome in ("srnet", "srnet_cur"):
        return f3.SRNet(canais, compacta=False)
    if nome in ("effnet", "effnet_cur"):
        return _EffNet(canais)
    raise ValueError(f"detector desconhecido: {nome}")


def _aumentar(x):
    k = random.randrange(4)
    if k:
        x = torch.rot90(x, k, dims=(2, 3))
    return torch.flip(x, dims=(3,)) if random.random() < 0.5 else x


def _semear(semente):
    torch.manual_seed(semente); random.seed(semente); np.random.seed(semente)


@torch.no_grad()
def _pontuar(rede, capas, estego, idx, cfg):
    dev, amp = cfg["dispositivo"], cfg["usar_amp"] and cfg["dispositivo"] == "cuda"
    rede.eval()
    lote = f30._lote_por_tamanho(128, capas)
    sc, sl = [], []
    for parte in (capas, estego):
        for i in range(0, len(idx), lote):
            b = torch.as_tensor(idx[i:i + lote])
            with torch.autocast("cuda", enabled=amp):
                sc.append(rede(parte[b].to(dev).float() / 255.0).float().cpu())
    s = torch.cat(sc).numpy()
    return s, np.r_[np.zeros(len(idx)), np.ones(len(idx))]


def treinar_detector(nome, capas, estego, idx_tr, idx_va, idx_te, cfg, semente, log=print,
                     inicial=None, epocas=None, rotulo=None):
    _semear(semente)
    dev, amp = cfg["dispositivo"], cfg["usar_amp"] and cfg["dispositivo"] == "cuda"
    epocas = epocas or cfg["epocas"]
    rotulo = rotulo or nome
    rede = construir_detector(nome, cfg["canais"]).to(dev)
    if inicial is not None:
        rede.load_state_dict(inicial)
    lr = cfg["lr_fino"] if inicial is not None else cfg["lr"][nome]
    opt = torch.optim.AdamW(rede.parameters(), lr=lr, weight_decay=cfg["decaimento"])
    pl = cfg["pares_por_lote"]
    passos = epocas * (len(idx_tr) // pl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=passos, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    capas_dev, estego_dev = capas.to(dev), estego.to(dev)

    melhor = {"pe": 2.0, "epoca": 0, "estado": None}
    t0 = time.time()
    for epoca in range(1, epocas + 1):
        rede.train()
        perm = np.random.permutation(idx_tr)
        soma, n = 0.0, 0
        for i in range(0, len(perm) - pl + 1, pl):
            b = torch.as_tensor(perm[i:i + pl], device=dev)
            x = _aumentar(torch.cat([capas_dev[b], estego_dev[b]]).float() / 255.0)
            y = torch.cat([torch.zeros(pl, device=dev), torch.ones(pl, device=dev)])
            with torch.autocast("cuda", enabled=amp):
                perda = F.binary_cross_entropy_with_logits(rede(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(perda).backward(); scaler.step(opt); scaler.update(); sched.step()
            soma += perda.item(); n += 1
        pe_va, auc_va = pe_auc(*_pontuar(rede, capas, estego, idx_va, cfg))
        marca = ""
        if pe_va <= melhor["pe"]:
            melhor = {"pe": pe_va, "epoca": epoca,
                      "estado": {k: v.detach().cpu().clone() for k, v in rede.state_dict().items()}}
            marca = "  *"
        restam = (epocas - epoca) * (time.time() - t0) / epoca
        log(f"[{rotulo}] época {epoca:2d}  bce={soma / max(n, 1):.4f}  "
            f"val: P_E={pe_va:.4f}  AUC={auc_va:.4f}  restam ~{restam / 60:.0f} min{marca}")
    del capas_dev, estego_dev
    if idx_te is None:
        return {"pe_val": melhor["pe"], "epoca_escolhida": melhor["epoca"], "estado": melhor["estado"]}
    rede.load_state_dict(melhor["estado"])
    notas, rotulos = _pontuar(rede, capas, estego, idx_te, cfg)
    pe, auc = pe_auc(notas, rotulos)
    return {"pe": pe, "auc": auc, "pe_val": melhor["pe"], "epoca_escolhida": melhor["epoca"],
            **quarteto(notas, rotulos), "pretreinada": getattr(rede, "pretreinada", "n/a"),
            "estado": melhor["estado"], "notas": notas.astype(np.float32),
            "minutos": (time.time() - t0) / 60}



def _espelhar_arquivo(local, cfg):
    if not cfg.get("espelho"):
        return
    destino = os.path.join(cfg["espelho"], os.path.relpath(local, cfg["pasta_saida"]))
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    shutil.copy2(local, destino)


def _no_espelho_ou_local(rel, cfg):
    for pasta in (cfg.get("espelho"), cfg["pasta_saida"]):
        if pasta and os.path.exists(os.path.join(pasta, rel)):
            return os.path.join(pasta, rel)
    return None


def preparar_curriculo(nome, capas, idx_tr, idx_va, cfg, semente, log=print):
    cargas = cfg["curriculo"][nome]
    rel = os.path.join("curriculo", f"{nome}_{'-'.join(f'{c:g}' for c in cargas)}_"
                       f"{cfg['epocas_curriculo']}ep_s{semente}_{len(capas)}.pt")
    achado = _no_espelho_ou_local(rel, cfg)
    if achado:
        log(f"[curric] {nome} s{semente}: reaproveitado de {achado}")
        return torch.load(achado)
    estado = None
    for carga in cargas:
        estego = _lsbm(capas, carga)
        r = treinar_detector(nome, capas, estego, idx_tr, idx_va, None, cfg, semente, log,
                             inicial=estado, epocas=cfg["epocas_curriculo"],
                             rotulo=f"{nome}@LSBM{carga:g} s{semente}")
        estado = r["estado"]
        aviso = "  <- NÃO APRENDEU (P_E ~0,5)" if r["pe_val"] > 0.45 else ""
        log(f"[curric] {nome} s{semente}: LSBM {carga:g} bpp -> P_E val {r['pe_val']:.4f}{aviso}")
    local = os.path.join(cfg["pasta_saida"], rel)
    os.makedirs(os.path.dirname(local), exist_ok=True)
    torch.save(estado, local)
    _espelhar_arquivo(local, cfg)
    return estado



COLUNAS = ["conjunto", "tamanho", "metodo", "bpp", "detector", "semente", "pe", "auc", "pe_val",
           "acuracia", "precisao", "recall", "f1", "epoca_escolhida", "pretreinada", "curriculo",
           "psnr", "ssim", "mse", "taxa", "razao_taxa_hill", "rms", "epocas", "minutos",
           "n_treino", "n_val", "n_teste"]


def _ler_tabela(caminho):
    if not caminho or not os.path.exists(caminho):
        return []
    with open(caminho, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _chave(conj, tam, metodo, bpp, det, semente):
    return (conj, int(tam), metodo, f"{float(bpp):g}", det, int(semente))


def nome_notas(metodo, bpp, det, semente):
    return os.path.join("notas", f"{metodo}_{bpp:g}_{det}_s{semente}.npy")


def rodar(metodos, cargas=(0.4, 0.2), detectores=("srnet_cur", "effnet"), sementes=(1234, 1235),
          cfg=None):
    cfg = dict(CONFIG, **(cfg or {}))
    os.makedirs(cfg["pasta_saida"], exist_ok=True)
    caminho_log = os.path.join(cfg["pasta_saida"], "fase40.log")
    t_inicio = time.time()

    def log(msg):
        print(msg)
        with open(caminho_log, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log_espelho = _no_espelho_ou_local("fase40.log", cfg)
    if log_espelho and log_espelho != caminho_log and not os.path.exists(caminho_log):
        shutil.copy2(log_espelho, caminho_log)

    tabela_local = os.path.join(cfg["pasta_saida"], "resultados_fase40.csv")
    linhas = _ler_tabela(_no_espelho_ou_local("resultados_fase40.csv", cfg))
    ja = {_chave(l["conjunto"], l["tamanho"], l["metodo"], l["bpp"], l["detector"], l["semente"])
          for l in linhas if int(l.get("epocas", 0)) == cfg["epocas"]}

    caminhos, idx_tr, idx_va, idx_te = dividir(cfg)
    log(f"[split ] {cfg['conjunto']} {cfg['tamanho_img']}²: {len(idx_tr)} treino_det / "
        f"{len(idx_va)} val_det / {len(idx_te)} teste (= val do gerador da Fase 3.0)")
    capas = carregar_capas(caminhos, cfg, log)
    np.save(os.path.join(cfg["pasta_saida"], "idx_teste.npy"), idx_te)
    _espelhar_arquivo(os.path.join(cfg["pasta_saida"], "idx_teste.npy"), cfg)

    nomes = [m["nome"] for m in metodos]
    assert "HILL" in nomes, "o HILL tem de estar entre os métodos: é a referência do ΔP_E"
    total = len(sementes) * len(cargas) * len(detectores) * len(metodos)
    feitos = sum(_chave(cfg["conjunto"], cfg["tamanho_img"], m, b, d, s) in ja
                 for s in sementes for b in cargas for d in detectores for m in nomes)
    log(f"[plano ] {total} treinos ({feitos} já na tabela): sementes {list(sementes)} × cargas "
        f"{list(cargas)} × detectores {list(detectores)} × métodos {nomes}")

    taxa_hill = {}
    for semente in sementes:
        for bpp in cargas:
            for det in detectores:
                pendentes = [m for m in metodos if _chave(cfg["conjunto"], cfg["tamanho_img"], m["nome"],
                                                          bpp, det, semente) not in ja]
                if not pendentes:
                    continue
                if cfg["limite_horas"] and time.time() - t_inicio > cfg["limite_horas"] * 3600:
                    log(f"[parada] limite de {cfg['limite_horas']} h: rode de novo para continuar")
                    return linhas
                inicial = None
                if det in cfg["curriculo"]:
                    inicial = preparar_curriculo(det, capas, idx_tr, idx_va, cfg, semente, log)
                for m in pendentes:
                    if bpp not in taxa_hill:
                        _, ih = gerar_estego(next(x for x in metodos if x["nome"] == "HILL"),
                                             capas, idx_te, bpp, cfg, log)
                        taxa_hill[bpp] = ih["taxa"]
                    estego, info = gerar_estego(m, capas, idx_te, bpp, cfg, log)
                    razao = info["taxa"] / taxa_hill[bpp]
                    if razao > cfg["razao_taxa_max"]:
                        log(f"[ALERTA] {m['nome']} @ {bpp:g} altera {razao:.3f}× o HILL: confira antes de ler o P_E")
                    log("=" * 78)
                    log(f"FASE 4.0 - {m['nome']} @ {bpp:g} bpp × {det} (semente {semente})")
                    log("=" * 78)
                    r = treinar_detector(det, capas, estego, idx_tr, idx_va, idx_te, cfg, semente, log,
                                         inicial=inicial, rotulo=f"{m['nome']}@{bpp:g} {det} s{semente}")
                    estado, notas = r.pop("estado"), r.pop("notas")
                    rel = nome_notas(m["nome"], bpp, det, semente)
                    os.makedirs(os.path.join(cfg["pasta_saida"], "notas"), exist_ok=True)
                    np.save(os.path.join(cfg["pasta_saida"], rel), notas)
                    _espelhar_arquivo(os.path.join(cfg["pasta_saida"], rel), cfg)
                    if cfg["salvar_pesos"]:
                        p = os.path.join(cfg["pasta_saida"], "pesos",
                                         f"{m['nome']}_{bpp:g}_{det}_s{semente}.pt")
                        os.makedirs(os.path.dirname(p), exist_ok=True)
                        torch.save(estado, p)
                        _espelhar_arquivo(p, cfg)
                        os.remove(p)
                    cur = cfg["curriculo"].get(det)
                    linha = {"conjunto": cfg["conjunto"], "tamanho": cfg["tamanho_img"], "metodo": m["nome"],
                             "bpp": f"{bpp:g}", "detector": det, "semente": semente, **r, **info,
                             "razao_taxa_hill": razao,
                             "curriculo": ">".join(f"{c:g}" for c in cur) if cur else "",
                             "epocas": cfg["epocas"], "n_treino": len(idx_tr), "n_val": len(idx_va),
                             "n_teste": len(idx_te)}
                    linhas.append({k: linha.get(k, "") for k in COLUNAS})
                    with open(tabela_local, "w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=COLUNAS); w.writeheader(); w.writerows(linhas)
                    _espelhar_arquivo(tabela_local, cfg)
                    log(f"[teste ] {m['nome']} @ {bpp:g} × {det} s{semente}: P_E={r['pe']:.4f}  "
                        f"AUC={r['auc']:.4f}  (época {r['epoca_escolhida']}, {r['minutos']:.0f} min)")
                    _espelhar_arquivo(caminho_log, cfg)
    log("[fim   ] todas as combinações medidas")
    _espelhar_arquivo(caminho_log, cfg)
    return linhas



def _bootstrap_delta(notas_a, notas_h, n_teste, reamostras, rng, grupos=None):
    y = np.r_[np.zeros(n_teste), np.ones(n_teste)]
    membros = None
    if grupos is not None:
        grupos = np.asarray(grupos)
        assert len(grupos) == n_teste, f"grupos ({len(grupos)}) ≠ imagens de teste ({n_teste})"
        membros = [np.flatnonzero(grupos == g) for g in np.unique(grupos)]
    out = np.empty(reamostras)
    for k in range(reamostras):
        if membros is None:
            i = rng.integers(0, n_teste, n_teste)
        else:
            i = np.concatenate([membros[g] for g in rng.integers(0, len(membros), len(membros))])
        j = np.r_[i, i + n_teste]
        out[k] = pe_auc(notas_a[j], y)[0] - pe_auc(notas_h[j], y)[0]
    return out


def grupos_por_cena(cfg):
    cfg = dict(CONFIG, **(cfg or {}))
    sep = f30.CONFIG_SPLIT.get("separador_cena")
    caminhos, _, _, idx_te = dividir(cfg)
    nomes = [os.path.basename(caminhos[i]) for i in idx_te]
    if not sep or not all(sep in n for n in nomes):
        return None
    return np.array([n.split(sep)[0] for n in nomes])


def _ic_pareado(espelho, met_a, met_b, bpp, det, sementes, reamostras, rng, grupos):
    boots = []
    for s in sementes:
        na = _carregar_notas(espelho, met_a, bpp, det, s)
        nb = _carregar_notas(espelho, met_b, bpp, det, s)
        if na is not None and nb is not None:
            boots.append(_bootstrap_delta(na, nb, len(na) // 2, reamostras, rng, grupos))
    if not boots:
        return float("nan"), float("nan")
    b = np.mean(boots, axis=0)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def resumo(espelho, reamostras=1000, semente=0, cenas=None, comparar=(("hibrido", "so_bce"),)):
    linhas = _ler_tabela(os.path.join(espelho, "resultados_fase40.csv"))
    if not linhas:
        print("tabela vazia"); return []
    rng = np.random.default_rng(semente)
    grupos = {}
    for l in linhas:
        grupos.setdefault((l["conjunto"], int(l["tamanho"]), l["bpp"], l["detector"], l["metodo"]), []).append(l)
    saida = []
    print("=" * 104)
    print("FASE 4.0 - P_E no teste (0,5 = indetectável). ΔP_E > 0 = mais seguro que o HILL")
    print("=" * 104)
    print(f"{'conj':>5}{'tam':>5}{'bpp':>6}{'detector':>11}{'método':>10}{'n':>3}{'P_E':>9}{'dp':>7}"
          f"{'ΔP_E':>9}{'IC 95%':>18}{'PSNR':>8}{'SSIM':>9}{'taxa':>8}")
    for chave in sorted(grupos, key=lambda c: (c[0], c[1], -float(c[2]), c[3], c[4] != "HILL", c[4])):
        conj, tam, bpp, det, met = chave
        g = grupos[chave]
        pes = np.array([float(l["pe"]) for l in g])
        ref = {int(l["semente"]): l for l in grupos.get((conj, tam, bpp, det, "HILL"), [])}
        delta, ic = float("nan"), (float("nan"), float("nan"))
        if met != "HILL":
            pares = [(l, ref[int(l["semente"])]) for l in g if int(l["semente"]) in ref]
            if pares:
                delta = float(np.mean([float(a["pe"]) - float(h["pe"]) for a, h in pares]))
                boots = []
                for a, h in pares:
                    na = _carregar_notas(espelho, met, bpp, det, a["semente"])
                    nh = _carregar_notas(espelho, "HILL", bpp, det, a["semente"])
                    if na is not None and nh is not None:
                        boots.append(_bootstrap_delta(na, nh, int(a["n_teste"]), reamostras, rng, cenas))
                if boots:
                    b = np.mean(boots, axis=0)
                    ic = (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))
        l0 = g[0]
        dp = float(pes.std(ddof=1)) if len(pes) > 1 else float("nan")
        r = {"conjunto": conj, "tamanho": tam, "bpp": bpp, "detector": det, "metodo": met,
             "n_sementes": len(pes), "pe_media": float(pes.mean()), "pe_dp": dp, "delta_pe": delta,
             "ic_baixo": ic[0], "ic_alto": ic[1], "psnr": float(l0["psnr"]),
             "ssim": float(l0["ssim"]), "taxa": float(l0["taxa"])}
        saida.append(r)
        ic_txt = "" if met == "HILL" or ic[0] != ic[0] else f"[{ic[0]:+.3f}, {ic[1]:+.3f}]"
        d_txt = "" if met == "HILL" else f"{delta:+.4f}"
        dp_txt = "" if dp != dp else f"{dp:.4f}"
        print(f"{conj:>5}{tam:>5}{float(bpp):>6.2f}{det:>11}{met:>10}{len(pes):>3}{pes.mean():>9.4f}"
              f"{dp_txt:>7}{d_txt:>9}{ic_txt:>18}{r['psnr']:>8.2f}{r['ssim']:>9.5f}{r['taxa']:>8.2%}")
    print("=" * 104)
    print("Leitura: diferença só conta se o IC 95% do ΔP_E não cruzar 0 (e, de preferência, nos dois detectores).")
    print("Bootstrap:", "por CENA" if cenas is not None else "por imagem")
    for a, b in comparar:
        celulas = sorted({k[:4] for k in grupos if k[4] == a and k[:4] + (b,) in grupos},
                         key=lambda c: (c[0], c[1], -float(c[2]), c[3]))
        if not celulas:
            continue
        print(f"\n{a} − {b} (IC 95% pareado direto; > 0 = {a} mais seguro)")
        for conj, tam, bpp, det in celulas:
            la = {int(l["semente"]): float(l["pe"]) for l in grupos[(conj, tam, bpp, det, a)]}
            lb = {int(l["semente"]): float(l["pe"]) for l in grupos[(conj, tam, bpp, det, b)]}
            sem = sorted(set(la) & set(lb))
            d = float(np.mean([la[s] - lb[s] for s in sem]))
            ic = _ic_pareado(espelho, a, b, bpp, det, sem, reamostras, rng, cenas)
            print(f"{conj:>5}{tam:>5}{float(bpp):>6.2f}{det:>11}  n={len(sem)}  {d:+.4f}  [{ic[0]:+.3f}, {ic[1]:+.3f}]")
            saida.append({"conjunto": conj, "tamanho": tam, "bpp": bpp, "detector": det,
                          "metodo": f"{a}-{b}", "n_sementes": len(sem), "pe_media": float("nan"),
                          "pe_dp": float("nan"), "delta_pe": d, "ic_baixo": ic[0], "ic_alto": ic[1],
                          "psnr": float("nan"), "ssim": float("nan"), "taxa": float("nan")})
    return saida


def _carregar_notas(espelho, metodo, bpp, det, semente):
    c = os.path.join(espelho, nome_notas(metodo, float(bpp), det, int(semente)))
    return np.load(c) if os.path.exists(c) else None
''')

open('fase46_baselines.py', 'w', encoding='utf-8').write(r'''import torch
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
''')

import importlib, fase0_codificador_decodificador, fase25_residual
importlib.reload(fase0_codificador_decodificador); importlib.reload(fase25_residual)
print('módulos regravados')

PARTES = ['classicos', 'profundo', 'boss256', 'utgan']
SEMENTES = [1234, 1235]
SEMENTES_256 = [1234]
CARGAS = [0.4, 0.2]
CARGA_BAIXA = 0.0625
DETECTORES = ['srnet_cur']
EPOCAS = 60
EPOCAS_UTGAN = 30
PARES_POR_LOTE = 32
LIMITE_HORAS = None

import os
from google.colab import drive
drive.mount('/content/drive')
assert os.path.ismount('/content/drive'), 'o Drive NÃO montou: PARE'
BASE = '/content/drive/MyDrive/IC_stego'
assert os.path.isdir(BASE), f'Drive montado, mas {BASE} não existe. PARE e confira o caminho.'
PASTA_DADOS_BASE = os.path.join(BASE, 'datasets')
PASTA_SAIDA_BASE = os.path.join(BASE, 'runs')
SAIDA = {128: os.path.join(PASTA_SAIDA_BASE, 'phase40', 'boss_longo60'),
         256: os.path.join(PASTA_SAIDA_BASE, 'phase40', 'boss_t256_longo60')}
for t, p in SAIDA.items():
    os.makedirs(p, exist_ok=True)
    tem = os.path.exists(os.path.join(p, 'resultados_fase40.csv'))
    print(f'{t}²: {p}', '(tabela OK)' if tem else '(SEM tabela: tudo será treinado)')
ESPELHO_30 = os.path.join(PASTA_SAIDA_BASE, 'phase30', 'boss', 'hibrido')
CKPT_HIB = os.path.join(ESPELHO_30, 'hibrido', 'ckpt_last.pt')
assert os.path.exists(CKPT_HIB), f'ckpt do hibrido não encontrado em {CKPT_HIB}: PARE'
PROFUNDOS = []
for nome, pasta, arq in (('stegs2gan', 'phase2_residual', 'residual'), ('denso', 'phase2_bossbase', 'denso')):
    ck = os.path.join(PASTA_SAIDA_BASE, pasta, 'ckpt_best.pt')
    if os.path.exists(ck):
        PROFUNDOS.append({'nome': nome, 'tipo': 'profundo', 'ckpt': ck, 'arquitetura': arq})
    print(f'ocultação profunda {nome}:', ck if os.path.exists(ck) else 'SEM ckpt (pulado)')

import glob, shutil, importlib, json, csv
import torch
import fase30_custo_wgan as f30
import fase40_pe as f40
import fase46_baselines as f46
importlib.reload(f30); importlib.reload(f40); importlib.reload(f46)
origem = os.path.join(PASTA_DADOS_BASE, 'BOSSbase_1.01')
PASTA = '/content/datasets/BOSSbase_1.01'
n = len(glob.glob(os.path.join(origem, '*.pgm')))
assert n > 0, f'BOSSBase não encontrado em {origem}'
if len(glob.glob(PASTA + '/*.pgm')) < n:
    shutil.copytree(origem, PASTA, dirs_exist_ok=True)

def cfg_de(tam):
    sub = 'boss_longo60' if tam == 128 else 'boss_t256_longo60'
    return dict(pasta_dados=PASTA, conjunto='boss', canais=1, tamanho_img=tam, epocas=EPOCAS,
                pares_por_lote=PARES_POR_LOTE, limite_horas=LIMITE_HORAS,
                pasta_saida=f'/content/runs/phase40/{sub}', espelho=SAIDA[tam])

HILL = {'nome': 'HILL', 'tipo': 'hill'}
HIB = {'nome': 'hibrido', 'tipo': 'custo', 'ckpt': CKPT_HIB}
UNIF = {'nome': 'unif', 'tipo': 'unif'}
SUNI = {'nome': 'suniward', 'tipo': 'suniward'}
LSBR = {'nome': 'lsb_subst', 'tipo': 'lsbr'}

cam, _, _, idx_te = f40.dividir(dict(f40.CONFIG, **cfg_de(128)))
from fase0_codificador_decodificador import DatasetImagens
ds = DatasetImagens([cam[i] for i in idx_te[:8]], 1, 128, treino=False)
x255 = torch.stack([torch.round(ds[i] * 255.0) for i in range(len(ds))]).cuda().double()
rng = torch.Generator().manual_seed(0)
rp, rm = f46.custo_suniward(x255)
y = f46.estego_suniward(x255, 0.4, rng)
taxa_s = (y != x255).double().mean().item()
taxa_h = (f30.estego_hill(x255, 0.4, rng) != x255).double().mean().item()
taxa_l = (f46.estego_lsbr(x255, 0.4, rng) != x255).double().mean().item()
finito = rp[rp < f30.CUSTO_MOLHADO]
assert torch.isfinite(rp).all() and finito.min() > 0, 'custo do S-UNIWARD inválido: PARE'
assert 0.5 < taxa_s / taxa_h < 2.0, f'S-UNIWARD altera {taxa_s:.2%} contra {taxa_h:.2%} do HILL: PARE'
assert abs(taxa_l - 0.2) < 0.01, f'LSB por substituição a 0,4 deveria alterar ~20%, alterou {taxa_l:.2%}: PARE'
print(f'sanidade a 0,4 bpp: S-UNIWARD altera {taxa_s:.2%} (HILL {taxa_h:.2%}), custo mediano {finito.median():.2f} | '
      f'LSB por substituição altera {taxa_l:.2%}')

if 'classicos' in PARTES:
    f40.rodar([HILL, HIB, SUNI, LSBR, UNIF], cargas=CARGAS, detectores=DETECTORES, sementes=SEMENTES, cfg=cfg_de(128))

if 'profundo' in PARTES:
    f40.rodar([HILL, HIB, UNIF, LSBR] + PROFUNDOS, cargas=[CARGA_BAIXA], detectores=DETECTORES, sementes=SEMENTES,
              cfg=cfg_de(128))

if 'boss256' in PARTES:
    f40.rodar([HILL, HIB, UNIF], cargas=[CARGA_BAIXA], detectores=DETECTORES, sementes=SEMENTES_256, cfg=cfg_de(256))

UTGAN_DIR = os.path.join(ESPELHO_30, 'utgan')
CKPT_UT = os.path.join(UTGAN_DIR, 'ckpt_last.pt')

def epocas_ut():
    if not os.path.exists(CKPT_UT):
        return 0
    return torch.load(CKPT_UT, map_location='cpu', weights_only=False)['epoca']

if 'utgan' in PARTES:
    est = json.load(open(os.path.join(ESPELHO_30, 'esteganalisador.json')))
    if epocas_ut() < EPOCAS_UTGAN:
        f30.treinar(dict(pasta_dados=PASTA, canais=1, tamanho_img=128, bpp=0.4, epocas=EPOCAS_UTGAN,
                         base='uniforme', adversario='bce', norma_critico='batch',
                         critico_inicial=est['arquivo_bn'], critico_wgan_inicial=est['arquivo'],
                         lambda_wgan=0.0, lambda_adv=1.0, lambda_z=0.0, lambda_drift=1e-3,
                         pasta_saida='/content/runs/phase30/boss/hibrido/utgan', espelho=UTGAN_DIR))
    assert epocas_ut() >= EPOCAS_UTGAN, f'gerador UT-GAN parou na época {epocas_ut()}: rode de novo para continuar'
    c = os.path.join(UTGAN_DIR, 'metricas.csv')
    if os.path.exists(c):
        ls = list(csv.DictReader(open(c, encoding='utf-8')))[-5:]
        m = lambda k: sum(float(l[k]) for l in ls) / len(ls)
        print(f"UT-GAN (5 últimas épocas): AUC do laço {m('auc_laco'):.3f} × HILL {m('auc_laco_hill'):.3f} | "
              f"taxa {m('taxa'):.2%} × HILL {m('taxa_hill'):.2%}")
    UT = {'nome': 'utgan', 'tipo': 'custo', 'ckpt': CKPT_UT}
    f40.rodar([HILL, HIB, UT], cargas=CARGAS, detectores=DETECTORES, sementes=SEMENTES, cfg=cfg_de(128))

importlib.reload(f40)
for tam in (128, 256):
    print(f'\n#################### BOSS {tam}² ####################')
    r = f40.resumo(SAIDA[tam], comparar=(('hibrido', 'suniward'), ('hibrido', 'utgan'), ('suniward', 'HILL'),
                                         ('hibrido', 'stegs2gan'), ('stegs2gan', 'unif'), ('hibrido', 'unif')))
    if r:
        with open(os.path.join(SAIDA[tam], 'resumo_fase46.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(r[0])); w.writeheader(); w.writerows(r)
    linhas = f40._ler_tabela(os.path.join(SAIDA[tam], 'resultados_fase40.csv'))
    pe = lambda met: [float(l['pe']) for l in linhas if l['metodo'] == met and float(l['bpp']) == CARGA_BAIXA]
    if pe('unif'):
        u = sum(pe('unif')) / len(pe('unif'))
        print(f"\ncontrole a {CARGA_BAIXA:g} em {tam}²: ±1 com P_E {u:.3f} -> "
              + ('régua TEM poder nessa carga: P_E ~0,5 de HILL/hibrido é resultado real' if u < 0.45 else
                 'régua SEM poder nessa carga (±1 também passa): célula inválida, comparar métodos a 0,2'))
for nome in [p['nome'] for p in PROFUNDOS] + ['lsb_subst']:
    for tam in (128,):
        for l in f40._ler_tabela(os.path.join(SAIDA[tam], 'resultados_fase40.csv')):
            if l['metodo'] == nome and l['semente'] == str(SEMENTES[0]):
                print(f"{nome} @ {l['bpp']}: PSNR {float(l['psnr']):.2f}  taxa {float(l['taxa']):.2%}  P_E {float(l['pe']):.4f}")
