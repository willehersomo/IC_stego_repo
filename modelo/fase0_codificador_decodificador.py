import os
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
