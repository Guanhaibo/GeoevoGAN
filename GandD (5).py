import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from dataclasses import dataclass
from typing import Sequence, Union, List, Tuple


# -----------------------------
# Config
# -----------------------------
@dataclass
class GConfig:
    z_dim: int = 100
    K: int = 64
    rank: int = 16


cfg = GConfig()


# -----------------------------
# Init / layers
# -----------------------------
def _get_weight_to_init(m: nn.Module):
    if hasattr(m, "weight_orig") and m.weight_orig is not None:
        return m.weight_orig
    if hasattr(m, "weight") and m.weight is not None:
        return m.weight
    return None


def weights_init(m):
    classname = m.__class__.__name__.lower()
    w = _get_weight_to_init(m)

    if "conv" in classname or "linear" in classname:
        if w is not None:
            nn.init.normal_(w.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias.data)

    elif "batchnorm" in classname:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias.data)


def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))


def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)


class GLU(nn.Module):
    def forward(self, x):
        c = x.size(1)
        assert c % 2 == 0, "channels dont divide 2!"
        c = c // 2
        return x[:, :c] * torch.sigmoid(x[:, c:])


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            conv2d(ch_in, ch_out, 4, 1, 0, bias=False),
            Swish(),
            conv2d(ch_out, ch_out, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()
        self.init = nn.Sequential(
            convTranspose2d(nz, channel * 2, 4, 1, 0, bias=False),
            batchNorm2d(channel * 2),
            GLU(),
        )

    def forward(self, z):
        z = z.reshape(z.shape[0], -1, 1, 1)
        return self.init(z)


# -----------------------------
# Generator blocks
# -----------------------------
class Evo(nn.Module):
    def __init__(
        self,
        D: int,
        shifts: Sequence[int] = (1, 2),
        dw_depth: int = 4,
        gate_hidden_mul: float = 1.0,
        gamma_init: float = 1e-4,
        use_dot: bool = True,
        use_wedge: bool = True,
        proj_groups: int = 1,
        evo_dim: int = 1,
        wedge_scale: float = 0.1,
    ):
        super().__init__()
        self.D = D
        self.shifts = tuple(int(s) for s in shifts)
        self.use_dot = bool(use_dot)
        self.use_wedge = bool(use_wedge)
        self.evo_dim = int(evo_dim)

        assert self.use_dot or self.use_wedge
        assert self.evo_dim in (1, 2, 3)

        self.norm = nn.GroupNorm(1, self.D)
        self.det = nn.Conv2d(self.D, self.D, 1, 1, 0, bias=True)

        ctx_layers = []
        for _ in range(int(dw_depth)):
            ctx_layers.append(nn.Conv2d(self.D, self.D, 3, 1, 1, groups=self.D, bias=False))
            ctx_layers.append(nn.LeakyReLU(0.2, inplace=True))
        ctx_layers.append(nn.Conv2d(self.D, self.D, 1, 1, 0, bias=True))
        self.ctx = nn.Sequential(*ctx_layers)

        per_shift = (self.D if self.use_dot else 0) + (self.D if self.use_wedge else 0)
        in_proj = per_shift * len(self.shifts)
        self.proj = nn.Conv2d(in_proj, self.D, 1, 1, 0, bias=True, groups=int(proj_groups))

        gate_hid = max(16, int(self.D * gate_hidden_mul))
        self.gate = nn.Sequential(
            nn.Conv2d(self.D * 2, gate_hid, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(gate_hid, self.D, 1, 1, 0, bias=True),
        )

        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.wedge_scale = nn.Parameter(torch.tensor(float(wedge_scale)))
        self.gamma = nn.Parameter(torch.full((1, self.D, 1, 1), float(gamma_init)))

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = self.norm(x0)

        det = self.act(self.det(x))
        ctx = self.ctx(x)

        feats = []
        for s in self.shifts:
            if self.evo_dim == 1:
                ctx_s = torch.roll(ctx, shifts=s, dims=1)
                det_s = torch.roll(det, shifts=s, dims=1)
            else:
                pad = [0, 0, 0, 0]  # left, right, top, bottom
                if self.evo_dim == 2:
                    if s > 0:
                        pad[2], pad[3] = s, 0
                    else:
                        pad[2], pad[3] = 0, -s
                else:
                    if s > 0:
                        pad[0], pad[1] = s, 0
                    else:
                        pad[0], pad[1] = 0, -s

                ctx_pad = F.pad(ctx, pad, mode="reflect")
                det_pad = F.pad(det, pad, mode="reflect")

                if self.evo_dim == 2:
                    ctx_s = ctx_pad[:, :, pad[2]:pad[2] + ctx.shape[2], :]
                    det_s = det_pad[:, :, pad[2]:pad[2] + det.shape[2], :]
                else:
                    ctx_s = ctx_pad[:, :, :, pad[0]:pad[0] + ctx.shape[3]]
                    det_s = det_pad[:, :, :, pad[0]:pad[0] + det.shape[3]]

            if self.use_dot:
                feats.append(self.act(det * ctx_s))

            if self.use_wedge:
                wedge = (det * ctx_s) - (ctx * det_s)
                feats.append(self.wedge_scale * wedge)

        g = self.proj(torch.cat(feats, dim=1))
        a = torch.sigmoid(self.gate(torch.cat([x, g], dim=1)))
        h = det + a * g
        return x0 + self.gamma * h


class Star(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 2.0, gamma_init: float = 1e-4, kkk: int = 7):
        super().__init__()
        hidden = max(16, int(dim * mlp_ratio))

        self.dwconv = conv2d(
            dim, dim, kkk, 1, kkk // 2,
            groups=dim, bias=False, padding_mode="reflect"
        )
        self.norm = batchNorm2d(dim)

        self.f1 = conv2d(dim, hidden, 1, 1, 0, bias=True)
        self.f2 = conv2d(dim, hidden, 1, 1, 0, bias=True)

        self.proj = nn.Sequential(
            conv2d(hidden, dim, 1, 1, 0, bias=False),
            batchNorm2d(dim),
        )

        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        h = self.dwconv(x)
        h = self.norm(h)
        x1 = self.f1(h)
        x2 = self.f2(h)
        h = self.proj(self.act(x1) * x2)
        return x0 + self.gamma * h


class UpBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        mlp_ratio: float = 2.0,
        use_evo: bool = True,
        star_gamma_init: float = 1e-4,
        evo_gamma_init: float = 1e-4,
        evo_shifts=(1, 2),
        evo_dw_depth: int = 4,
        wedge: bool = True,
        kkk: int = 7,
        block_res_scale: float = 0.5,
        evo_dims=(1,),
    ):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        if in_ch != out_ch:
            self.proj = nn.Sequential(
                conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
                batchNorm2d(out_ch),
            )
        else:
            self.proj = nn.Identity()

        self.star = Star(
            dim=out_ch,
            mlp_ratio=mlp_ratio,
            gamma_init=star_gamma_init,
            kkk=kkk,
        )

        if use_evo and len(evo_dims) > 0:
            evos = []
            for d in evo_dims:
                evos.append(
                    Evo(
                        D=out_ch,
                        shifts=evo_shifts,
                        dw_depth=evo_dw_depth,
                        gate_hidden_mul=1.0,
                        gamma_init=evo_gamma_init,
                        use_dot=True,
                        use_wedge=wedge,
                        proj_groups=1,
                        evo_dim=d,
                        wedge_scale=0.1,
                    )
                )
            self.evos = nn.Sequential(*evos)
        else:
            self.evos = nn.Identity()

        self.block_res_scale = nn.Parameter(torch.tensor(float(block_res_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.up(x)
        t = self.proj(t)
        h = self.star(t)
        h = self.evos(h)
        return t + self.block_res_scale * (h - t)


class QFiLM(nn.Module):
    def __init__(self, K: int, C: int, r: int, hidden_mul: float = 2.0, gamma_scale: float = 0.1):
        super().__init__()
        self.K = K
        self.C = C
        self.r = r
        self.gamma_scale = gamma_scale

        in_dim = K * r
        hid = max(32, int(in_dim * hidden_mul))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hid, 2 * C),
        )

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        assert c == self.C

        s = q.reshape(b, self.K * self.r)
        gb = self.mlp(s)
        gamma, beta = gb[:, :c], gb[:, c:]

        gamma = (self.gamma_scale * gamma).view(b, c, 1, 1)
        beta = beta.view(b, c, 1, 1)

        return x * (1.0 + gamma) + beta


class QR(nn.Module):
    def __init__(self, z_dim: int, K: int, rank: int):
        super().__init__()
        self.K = K
        self.rank = rank
        self.fc = nn.Linear(z_dim, K * rank)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        w = self.fc(z).view(b, self.K, self.rank)
        w = w + 1e-6
        q, _ = torch.linalg.qr(w.float(), mode="reduced")
        return q.to(z.dtype)


# -----------------------------
# Generator
# -----------------------------
class Generator(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=256):
        super().__init__()

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5}
        nfc = {k: int(v * ngf) for k, v in nfc_multi.items()}

        assert im_size == 256, "this version is fixed for 256"
        self.im_size = im_size

        self.init = InitLayer(nz, channel=nfc[4])

        self.feat_8 = UpBlock(
            nfc[4], nfc[8],
            kkk=3,
            evo_dims=(1, 1),
            wedge=True,
            block_res_scale=0.5,
        )

        self.feat_16 = UpBlock(
            nfc[8], nfc[16],
            kkk=5,
            evo_dims=(1, 2),
            wedge=True,
            block_res_scale=0.45,
        )

        self.feat_32 = UpBlock(
            nfc[16], nfc[32],
            kkk=7,
            evo_dims=(1, 3),
            wedge=True,
            block_res_scale=0.4,
        )

        self.feat_64 = UpBlock(
            nfc[32], nfc[64],
            kkk=7,
            evo_dims=(1, 2, 3),
            wedge=True,
            block_res_scale=0.35,
        )

        self.feat_128 = UpBlock(
            nfc[64], nfc[128],
            kkk=3,
            evo_dims=(1,),
            wedge=False,
            block_res_scale=0.3,
        )

        self.feat_256 = UpBlock(
            nfc[128], nfc[256],
            kkk=3,
            use_evo=False,
            wedge=False,
            block_res_scale=0.25,
        )

        self.to_128 = nn.Conv2d(nfc[128], nc, 1, 1, 0, bias=False)
        self.to_big = nn.Conv2d(nfc[256], nc, 3, 1, 1, bias=False, padding_mode="reflect")

        self.seed = QR(cfg.z_dim, cfg.K, cfg.rank)
        self.qfilm_32 = QFiLM(K=cfg.K, C=nfc[32], r=cfg.rank, gamma_scale=0.1)
        self.qfilm_128 = QFiLM(K=cfg.K, C=nfc[128], r=cfg.rank, gamma_scale=0.1)

    def forward(self, z):
        q = self.seed(z)

        feat = self.init(z)
        feat = self.feat_8(feat)
        feat_16 = self.feat_16(feat)

        feat_32 = self.feat_32(feat_16)
        feat_32 = self.qfilm_32(feat_32, q)

        feat_64 = self.feat_64(feat_32)

        feat_128 = self.feat_128(feat_64)
        feat_128 = self.qfilm_128(feat_128, q)

        feat_256 = self.feat_256(feat_128)

        return [
            torch.tanh(self.to_big(feat_256)),
            torch.tanh(self.to_128(feat_128)),
        ]


# -----------------------------
# Discriminator blocks
# -----------------------------
class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.main(x)


class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
            conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
            batchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2),
            conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            batchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return 0.5 * (self.main(x) + self.direct(x))


class SimpleDecoder(nn.Module):
    def __init__(self, nfc_in=64, nc=3):
        super().__init__()

        nfc_multi = {
            4: 16, 8: 8, 16: 4, 32: 2, 64: 2,
            128: 1, 256: 0.5, 512: 0.25, 1024: 0.125
        }
        nfc = {k: int(v * 32) for k, v in nfc_multi.items()}

        def up_block(in_planes, out_planes):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
                batchNorm2d(out_planes * 2),
                GLU(),
            )

        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            up_block(nfc_in, nfc[16]),
            up_block(nfc[16], nfc[32]),
            up_block(nfc[32], nfc[64]),
            up_block(nfc[64], nfc[128]),
            conv2d(nfc[128], nc, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x)


# -----------------------------
# Discriminator
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=256):
        super().__init__()
        self.ndf = ndf
        self.im_size = im_size

        nfc_multi = {
            4: 16, 8: 16, 16: 8, 32: 4, 64: 2,
            128: 1, 256: 0.5, 512: 0.25, 1024: 0.125
        }
        nfc = {k: int(v * ndf) for k, v in nfc_multi.items()}

        if im_size == 1024:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[1024], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(nc if False else nfc[1024], nfc[512], 4, 2, 1, bias=False),
                batchNorm2d(nfc[512]),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif im_size == 512:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif im_size == 256:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            raise ValueError(f"Unsupported im_size: {im_size}")

        self.down_4 = DownBlockComp(nfc[512], nfc[256])
        self.down_8 = DownBlockComp(nfc[256], nfc[128])
        self.down_16 = DownBlockComp(nfc[128], nfc[64])
        self.down_32 = DownBlockComp(nfc[64], nfc[32])
        self.down_64 = DownBlockComp(nfc[32], nfc[16])

        self.rf_big = nn.Sequential(
            conv2d(nfc[16], nfc[8], 1, 1, 0, bias=False),
            batchNorm2d(nfc[8]),
            nn.LeakyReLU(0.2, inplace=True),
            conv2d(nfc[8], 1, 4, 1, 0, bias=False),
        )

        self.se_2_16 = SEBlock(nfc[512], nfc[64])
        self.se_4_32 = SEBlock(nfc[256], nfc[32])
        self.se_8_64 = SEBlock(nfc[128], nfc[16])

        self.down_from_small = nn.Sequential(
            conv2d(nc, nfc[256], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            DownBlock(nfc[256], nfc[128]),
            DownBlock(nfc[128], nfc[64]),
            DownBlock(nfc[64], nfc[32]),
        )

        self.rf_small = conv2d(nfc[32], 1, 4, 1, 0, bias=False)

        self.decoder_big = SimpleDecoder(nfc[16], nc)
        self.decoder_small = SimpleDecoder(nfc[32], nc)
        self.decoder_part = SimpleDecoder(nfc[32], nc)

    def _resize_img(self, img: torch.Tensor, size: int) -> torch.Tensor:
        if img.shape[-2:] == (size, size):
            return img
        return F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False)

    def _parse_inputs(self, imgs: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]):
        if isinstance(imgs, (list, tuple)):
            big = self._resize_img(imgs[0], self.im_size)
            small = self._resize_img(imgs[1], 128)
        else:
            big = self._resize_img(imgs, self.im_size)
            small = self._resize_img(imgs, 128)
        return big, small

    def _extract_features(self, imgs):
        big, small = self._parse_inputs(imgs)

        feat_2 = self.down_from_big(big)
        feat_4 = self.down_4(feat_2)
        feat_8 = self.down_8(feat_4)

        feat_16 = self.down_16(feat_8)
        feat_16 = self.se_2_16(feat_2, feat_16)

        feat_32 = self.down_32(feat_16)
        feat_32 = self.se_4_32(feat_4, feat_32)

        feat_last = self.down_64(feat_32)
        feat_last = self.se_8_64(feat_8, feat_last)

        feat_small = self.down_from_small(small)

        return feat_last, feat_small, feat_32

    def _score(self, feat_last, feat_small):
        rf_big = self.rf_big(feat_last).reshape(feat_last.size(0), -1).squeeze(1)
        rf_small = self.rf_small(feat_small).reshape(feat_small.size(0), -1).squeeze(1)
        return torch.cat([rf_big, rf_small], dim=0)

    def _part_feature(self, feat_32, part: int):
        h, w = feat_32.shape[2:]
        h2, w2 = h // 2, w // 2

        if part == 0:
            return feat_32[:, :, :h2, :w2]
        if part == 1:
            return feat_32[:, :, :h2, w2:]
        if part == 2:
            return feat_32[:, :, h2:, :w2]
        if part == 3:
            return feat_32[:, :, h2:, w2:]
        raise ValueError(f"Invalid part: {part}")

    def forward_logits(self, imgs):
        feat_last, feat_small, _ = self._extract_features(imgs)
        return self._score(feat_last, feat_small)

    def forward_recon(self, imgs, part: int):
        feat_last, feat_small, feat_32 = self._extract_features(imgs)

        rec_big = self.decoder_big(feat_last)
        rec_small = self.decoder_small(feat_small)
        rec_part = self.decoder_part(self._part_feature(feat_32, part))

        return [rec_big, rec_small, rec_part]

    def forward(self, imgs):
        return self.forward_logits(imgs)