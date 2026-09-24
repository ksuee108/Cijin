"""
DiffusionEdge 的条件 Unet -> UNet++ (嵌套密集跳跃连接) 改造版。

改动说明
--------
原始 `mask_cond_unet.Unet` 是标准 U-Net 结构：编码器每一层产生一个跳跃特征
h[i]，解码器在对应层直接 concat(h.pop(), x) 后卷积。

本文件把解码路径换成 UNet++ (Zhou et al., 2018) 的嵌套密集跳跃连接：

    X[i][0] = 编码器第 i 层输出（i = 0..L-2），
              X[L-1][0] = 中间层(bottleneck)特征与最深层编码特征融合后的投影
    X[i][j] = ConvBlock( concat( X[i][0], ..., X[i][j-1], Upsample(X[i+1][j-1]) ) )   for j >= 1

最终解码输出取最浅层、最右侧列 X[0][L-1]，与原代码一样再与 init_conv 的
输出 r 拼接后过 final_res_block/final_conv 得到结果。

保留的原始设计：
  * 编码器侧通过 RelationNet 做 mask 条件跨注意力（cond backbone: effnet/resnet/swin/vgg）。
  * 两条独立解码分支 x1 (预测 C) / x2 (预测 noise)，来自中间层 FFT decouple。
  * EDM 风格的 c_skip / c_out 预处理系数。

简化之处（如需还原请告诉我）：
  * UNet++ 的嵌套节点只在两条解码分支各自的路径中做纯卷积密集融合，
    不在每个嵌套节点重复注入 RelationNet 的 mask 条件跨注意力
    （否则参数量/显存会随嵌套层数平方增长）。
  * 未加入 UNet++ 常见的"深监督(deep supervision)"多输出平均，
    因为这个模型的扩散参数化（C_pred, noise_pred）没有现成的多尺度监督目标对应关系。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from denoising_diffusion_pytorch.efficientnet import efficientnet_b7, EfficientNet_B7_Weights
from denoising_diffusion_pytorch.resnet import resnet101, ResNet101_Weights
from denoising_diffusion_pytorch.swin_transformer import swin_b, Swin_B_Weights
from denoising_diffusion_pytorch.vgg import vgg16, VGG16_Weights

# 复用原文件里已经写好、且没有改动的组件
from denoising_diffusion_pytorch.mask_cond_unet import (
    Residual,
    PreNorm,
    LinearAttention,
    Attention,
    ResnetBlock,
    BlockFFT,
    Upsample,
    Downsample,
    RelationNet,
    GaussianFourierProjection,
    RandomOrLearnedSinusoidalPosEmb,
    exists,
    default,
)


class UnetPlusPlus(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        cond_in_dim=1,
        cond_dim=64,
        cond_dim_mults=(2, 4, 8),
        channels=1,
        out_mul=1,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        window_sizes1=[[16, 16], [8, 8], [4, 4], [2, 2]],
        window_sizes2=[[16, 16], [8, 8], [4, 4], [2, 2]],
        fourier_scale=16,
        ckpt_path=None,
        ignore_keys=[],
        cfg={},
        **kwargs
    ):
        super().__init__()

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)

        # ---------------- mask 条件主干网络 ----------------
        if cfg.cond_net == 'effnet':
            f_condnet = 48
            self.init_conv_mask = efficientnet_b7() if cfg.get('without_pretrain', False) \
                else efficientnet_b7(weights=EfficientNet_B7_Weights)
        elif cfg.cond_net == 'resnet':
            f_condnet = 256
            self.init_conv_mask = resnet101() if cfg.get('without_pretrain', False) \
                else resnet101(weights=ResNet101_Weights)
        elif cfg.cond_net == 'swin':
            f_condnet = 128
            self.init_conv_mask = swin_b() if cfg.get('without_pretrain', False) \
                else swin_b(weights=Swin_B_Weights)
        elif cfg.cond_net == 'vgg':
            f_condnet = 128
            self.init_conv_mask = vgg16() if cfg.get('without_pretrain', False) \
                else vgg16(weights=VGG16_Weights)
        else:
            raise NotImplementedError

        self.init_conv = nn.Sequential(
            nn.Conv2d(input_channels + f_condnet, init_dim, 7, padding=3),
            nn.GroupNorm(num_groups=min(init_dim // 4, 8), num_channels=init_dim),
        )

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        L = len(in_out)                    # 分辨率层数
        self.num_resolutions = L
        self.level_dims = dims[:-1]        # 每一层(i=0..L-1)的基础通道数 C_i
        mid_dim = dims[-1]

        # mask 特征在各层的投影（与原代码一致，仅编码阶段用于 RelationNet）
        self.projects = nn.ModuleList()
        print(cfg.cond_net)
        if cfg.cond_net == 'effnet':
            self.projects.append(nn.Conv2d(48, dims[0], 1))
            self.projects.append(nn.Conv2d(80, dims[1], 1))
            self.projects.append(nn.Conv2d(224, dims[2], 1))
            self.projects.append(nn.Conv2d(640, dims[3], 1))
        elif cfg.cond_net == 'vgg':
            self.projects.append(nn.Conv2d(128, dims[0], 1))
            self.projects.append(nn.Conv2d(256, dims[1], 1))
            self.projects.append(nn.Conv2d(512, dims[2], 1))
            self.projects.append(nn.Conv2d(512, dims[3], 1))
        else:
            self.projects.append(nn.Conv2d(f_condnet, dims[0], 1))
            self.projects.append(nn.Conv2d(f_condnet * 2, dims[1], 1))
            self.projects.append(nn.Conv2d(f_condnet * 4, dims[2], 1))
            self.projects.append(nn.Conv2d(f_condnet * 8, dims[3], 1))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # ---------------- 时间步嵌入 ----------------
        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = GaussianFourierProjection(dim // 2, scale=fourier_scale)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # ---------------- 编码器（与原来一致） ----------------
        self.downs = nn.ModuleList([])
        self.relation_layers_down = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (L - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))
            self.relation_layers_down.append(
                RelationNet(in_channel1=dims[ind], in_channel2=dims[ind], nhead=8,
                            layers=1, embed_dim=dims[ind], ffn_dim=dims[ind] * 2,
                            window_size1=window_sizes1[ind], window_size2=window_sizes2[ind])
            )

        # ---------------- 中间层（与原来一致，含 FFT decouple 双分支） ----------------
        input_size = cfg.get('input_size', [80, 80])
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.decouple1 = nn.Sequential(
            nn.GroupNorm(num_groups=min(mid_dim // 4, 8), num_channels=mid_dim),
            nn.Conv2d(mid_dim, mid_dim, 3, padding=1),
            BlockFFT(mid_dim, input_size[0] // 8, input_size[1] // 8),
        )
        self.decouple2 = nn.Sequential(
            nn.GroupNorm(num_groups=min(mid_dim // 4, 8), num_channels=mid_dim),
            nn.Conv2d(mid_dim, mid_dim, 3, padding=1),
            BlockFFT(mid_dim, input_size[0] // 8, input_size[1] // 8),
        )

        # 把 bottleneck 特征（mid_dim 通道）和最深层编码特征（dims[L-1] 通道）
        # 融合、投影成 UNet++ 网格最深一列 X[L-1][0]（dims[L-1] 通道）
        self.mid_proj1 = block_klass(mid_dim + dims[L - 1], dims[L - 1], time_emb_dim=time_dim)
        self.mid_proj2 = block_klass(mid_dim + dims[L - 1], dims[L - 1], time_emb_dim=time_dim)

        # ---------------- UNet++ 嵌套解码网格（两条分支各一份） ----------------
        self.ups1, self.blocks1 = self._build_nested_grid(self.level_dims, L, time_dim, block_klass)
        self.ups2, self.blocks2 = self._build_nested_grid(self.level_dims, L, time_dim, block_klass)

        # ---------------- 输出头（与原来一致） ----------------
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dims[0] * 2, dims[0], time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dims[0], self.out_dim * out_mul, 1)

        self.final_res_block2 = block_klass(dims[0] * 2, dims[0], time_emb_dim=time_dim)
        self.final_conv2 = nn.Conv2d(dims[0], self.out_dim, 1)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        fix_bb = cfg.get('fix_bb', True)
        if fix_bb:
            for n, p in self.init_conv_mask.named_parameters():
                p.requires_grad = False

    @staticmethod
    def _build_nested_grid(level_dims, L, time_dim, block_klass):
        """
        构建 UNet++ 的嵌套跳跃连接网格。
        对每个 (i, j)（i = 0..L-2, j = 1..L-1-i）：
          - ups[f"{i}_{j}"]  : 把 X[i+1][j-1] 上采样并投影到第 i 层的通道数
          - blocks[f"{i}_{j}"]: 把 concat(X[i][0..j-1], up_feat) 卷积回第 i 层通道数
        """
        ups = nn.ModuleDict()
        blocks = nn.ModuleDict()
        for i in range(L - 1):
            c_i = level_dims[i]
            c_ip1 = level_dims[i + 1]
            for j in range(1, L - i):
                key = f"{i}_{j}"
                ups[key] = Upsample(c_ip1, c_i)
                blocks[key] = block_klass((j + 1) * c_i, c_i, time_emb_dim=time_dim)
        return ups, blocks

    def _forward_nested_grid(self, enc_feats, bottom_feat, ups, blocks, t, L):
        """
        enc_feats  : list，长度 L-1，enc_feats[i] 对应 X[i][0]，i = 0..L-2
        bottom_feat: X[L-1][0]（已经融合了 bottleneck 与最深层编码特征）
        返回        : X[0][L-1]（解码路径最终、分辨率最高、最右列的特征）
        """
        X = {}
        for i in range(L - 1):
            X[(i, 0)] = enc_feats[i]
        X[(L - 1, 0)] = bottom_feat

        for i in range(L - 2, -1, -1):
            for j in range(1, L - i):
                up_feat = ups[f"{i}_{j}"](X[(i + 1, j - 1)])
                target_size = X[(i, 0)].shape[-2:]
                if up_feat.shape[-2:] != target_size:
                    up_feat = F.interpolate(up_feat, size=target_size, mode="bilinear", align_corners=True)
                cat_feat = torch.cat([X[(i, k)] for k in range(j)] + [up_feat], dim=1)
                X[(i, j)] = blocks[f"{i}_{j}"](cat_feat, t)

        return X[(0, L - 1)]

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["model"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        # 注意：解码路径结构（UNet++）与原 checkpoint（普通 U-Net）不同，
        # 编码器 / mask backbone / 中间层权重可以复用，解码器部分权重不会匹配，
        # 会以 missing/unexpected keys 的形式被跳过（strict=False）。
        msg = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        print('==>Load UnetPlusPlus Info: ', msg)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x, time, mask, x_self_cond=None, **kwargs):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        # ---- EDM 风格预处理系数（与原代码一致） ----
        sigma = time.reshape(-1, 1, 1, 1)
        c_skip1 = 1 - sigma
        c_skip2 = torch.sqrt(sigma)
        c_out1 = sigma / torch.sqrt(sigma ** 2 + 1)
        c_out2 = torch.sqrt(1 - sigma) / torch.sqrt(sigma ** 2 + 1)
        c_in = 1

        x_clone = x.clone()
        x = c_in * x

        hm = self.init_conv_mask(mask)
        x = self.init_conv(torch.cat([x, F.interpolate(hm[0], size=x.shape[-2:], mode="bilinear")], dim=1))
        r = x.clone()

        t = self.time_mlp(torch.log(time) / 4)

        for i, layer in enumerate(self.projects):
            hm[i] = layer(hm[i])

        # ---- 编码器：跟原来一样，用 RelationNet 融合 mask 条件 ----
        enc_feats = []
        for i, ((block1, block2, attn, downsample), relation_layer) \
                in enumerate(zip(self.downs, self.relation_layers_down)):
            x = block1(x, t)
            x = relation_layer(hm[i], x)
            x = block2(x, t)
            x = attn(x)
            enc_feats.append(x)
            x = downsample(x)

        # ---- 中间层 + FFT 双分支 decouple（与原来一致） ----
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        x1 = x + self.decouple1(x)
        x2 = x + self.decouple2(x)

        L = self.num_resolutions
        deepest_enc = enc_feats[L - 1]
        x1_bottom = self.mid_proj1(torch.cat([x1, deepest_enc], dim=1), t)
        x2_bottom = self.mid_proj2(torch.cat([x2, deepest_enc], dim=1), t)

        # ---- UNet++ 嵌套解码网格（两条分支分别计算） ----
        x1_top = self._forward_nested_grid(enc_feats[:L - 1], x1_bottom, self.ups1, self.blocks1, t, L)
        x2_top = self._forward_nested_grid(enc_feats[:L - 1], x2_bottom, self.ups2, self.blocks2, t, L)

        x1 = torch.cat((x1_top, r), dim=1)
        x1 = self.final_res_block(x1, t)
        x1 = self.final_conv(x1)

        x2 = torch.cat((x2_top, r), dim=1)
        x2 = self.final_res_block2(x2, t)
        x2 = self.final_conv2(x2)

        x1 = c_skip1 * x_clone + c_out1 * x1
        x2 = c_skip2 * x_clone + c_out2 * x2
        return x1, x2


if __name__ == "__main__":
    import fvcore.common.config

    model = UnetPlusPlus(
        dim=128, dim_mults=(1, 2, 4, 4),
        cond_dim=128,
        cond_dim_mults=(2, 4,),
        channels=1,
        window_sizes1=[[8, 8], [4, 4], [2, 2], [1, 1]],
        window_sizes2=[[8, 8], [4, 4], [2, 2], [1, 1]],
        cfg=fvcore.common.config.CfgNode({
            'cond_pe': False, 'input_size': [80, 80],
            'cond_feature_size': (32, 128), 'cond_net': 'vgg',
            'num_pos_feats': 96
        })
    )
    x = torch.rand(1, 1, 80, 80)
    mask = torch.rand(1, 3, 320, 320)
    time = torch.tensor([0.5124])
    with torch.no_grad():
        y = model(x, time, mask)
    print(y[0].shape, y[1].shape)
