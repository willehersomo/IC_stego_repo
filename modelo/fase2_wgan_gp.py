
import os
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import fase0_codificador_decodificador as base   # reusa arquitetura, loaders e métricas da Fase 0


CONFIG = dict(base.CONFIG)
CONFIG.update({
    "lambda_adv": 0.01,       
                               
    "n_critic": 5,             
    "lambda_gp": 10.0,         
    "largura_critico": 64,     
    "lr_critico": 1e-4,       
    "beta1": 0.0, "beta2": 0.9,  # betas do Adam (recomendado p/ WGAN-GP)

    "taxa_aprend": 1e-4,      
    "epocas": 60,
    "usar_amp": False,         # GP + AMP não combinam
    "carregar_fase0": "runs/phase0_bossbase/ckpt_best.pt",  # warm start; None = do zero
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
            camadas = [nn.Conv2d(entrada, saida, 4, stride=2, padding=1)]  # reduz H,W pela metade
            if norm:
                camadas.append(nn.InstanceNorm2d(saida, affine=True))
            camadas.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*camadas)

        self.corpo = nn.Sequential(
            bloco(canais, largura, norm=False),   # 1ª camada sem norm
            bloco(largura, largura * 2),
            bloco(largura * 2, largura * 4),
            bloco(largura * 4, largura * 8),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)          # vira (B, C, 1, 1) p/ qualquer tamanho de entrada
        self.para_score = nn.Linear(largura * 8, 1)  # score escalar por imagem

    def forward(self, x):
        h = self.corpo(x)
        h = self.pool(h).flatten(1)     # (B, C)
        return self.para_score(h).squeeze(1)  # (B,)


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

    # --- warm start da Fase 0 (colocar o adversário sobre uma base que já funciona) 
    ck = cfg.get("carregar_fase0")
    if ck and os.path.exists(ck):
        estado = torch.load(ck, map_location=dispositivo)
        codificador.load_state_dict(estado["codificador"])
        decodificador.load_state_dict(estado["decodificador"])
        print(f"[warm start] pesos da Fase 0 carregados de {ck}")
    else:
        print(f"[warm start] {ck!r} não encontrado -> treinando do zero "
              "(recomendado rodar a Fase 0 antes)")

    # gerador = codificador + decodificador (treinados juntos); crítico separado
    opt_g = torch.optim.Adam(list(codificador.parameters()) + list(decodificador.parameters()),
                             lr=cfg["taxa_aprend"], betas=(cfg["beta1"], cfg["beta2"]))
    opt_c = torch.optim.Adam(critico.parameters(),
                             lr=cfg["lr_critico"], betas=(cfg["beta1"], cfg["beta2"]))

    melhor = {"score": -1.0, "epoca": 0, "acuracia_bits": 0.0, "psnr": 0.0, "ssim": 0.0}

    for epoca in range(1, cfg["epocas"] + 1):
        codificador.train(); decodificador.train(); critico.train()
        ac = {"c": 0.0, "w": 0.0, "adv": 0.0, "img": 0.0, "msg": 0.0, "acc": 0.0, "n": 0}

        for portadora in loader_treino:
            portadora = portadora.to(dispositivo, non_blocking=True)
            b = portadora.size(0)

            # ============ 1) treina o CRÍTICO n_critic vezes ============
            for _ in range(cfg["n_critic"]):
                mensagem = base.mensagem_aleatoria(b, cfg)
                with torch.no_grad():                       # não propaga p/ o gerador aqui
                    estego = codificador(portadora, base.espalhar_mensagem(mensagem, cfg))
                score_real = critico(portadora).mean()
                score_falsa = critico(estego).mean()
                gp = penalidade_gradiente(critico, portadora, estego, dispositivo)
                # crítico MINIMIZA:  E[C(falsa)] - E[C(real)] + lambda_gp * GP
                perda_c = score_falsa - score_real + cfg["lambda_gp"] * gp
                opt_c.zero_grad(); perda_c.backward(); opt_c.step()

            w_dist = (score_real - score_falsa).item()      # estimativa da distância de Wasserstein

            # ============ 2) treina o GERADOR 1 vez ============
            mensagem = base.mensagem_aleatoria(b, cfg)
            estego = codificador(portadora, base.espalhar_mensagem(mensagem, cfg))
            logits = base.agregar_logits(decodificador(estego), cfg)

            perda_img = F.mse_loss(estego, portadora)
            perda_msg = F.binary_cross_entropy_with_logits(logits, mensagem)
            perda_res = (estego - portadora).abs().mean()
            perda_adv = -critico(estego).mean()             # gerador quer SUBIR o score da estego
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

        # --- validação (reusa a métrica da Fase 0) + amostras ---
        val = base.avaliar(codificador, decodificador, loader_val, cfg)
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
