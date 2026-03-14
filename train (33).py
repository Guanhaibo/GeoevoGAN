import os
import csv
import math
import random
import argparse

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.utils import save_image

from tqdm import tqdm
import matplotlib.pyplot as plt
import lpips

from dataset_poem import PoemImageDataset
from GandD import weights_init, Discriminator, Generator
from operation import get_dir, InfiniteSamplerWrapper
from diffaug import DiffAugment


# -----------------------------
# Global
# -----------------------------
policy = "color,translation"


# -----------------------------
# Schedulers / losses
# -----------------------------
def get_three_stage_scheduler(
    optimizer,
    total_iters,
    warmup_end=2000,
    decay_end=20000,
    peak_lr=5e-4,
    mid_lr=2e-4,
    min_lr_ratio=0.01,
    last_epoch=-1,
):
    """
    3-stage scheduler
    stage 1: [0, warmup_end)         0       -> peak_lr
    stage 2: [warmup_end, decay_end) peak_lr -> mid_lr
    stage 3: [decay_end, total_iters] cosine mid_lr -> mid_lr * min_lr_ratio
    """
    if total_iters <= decay_end:
        raise ValueError(f"total_iters ({total_iters}) must be > decay_end ({decay_end})")

    base_lr = optimizer.param_groups[0]["lr"]
    final_lr = mid_lr * min_lr_ratio

    if abs(base_lr - peak_lr) > 1e-12:
        print(
            f"[Warning] optimizer base lr = {base_lr}, but peak_lr = {peak_lr}. "
            f"Usually these should match."
        )

    def lr_lambda(step):
        step = max(step, 0)

        # stage 1: warmup
        if step < warmup_end:
            lr = peak_lr * (step / warmup_end)

        # stage 2: linear decay
        elif step < decay_end:
            progress = (step - warmup_end) / (decay_end - warmup_end)
            lr = peak_lr + (mid_lr - peak_lr) * progress

        # stage 3: cosine decay
        else:
            progress = (step - decay_end) / (total_iters - decay_end)
            progress = min(max(progress, 0.0), 1.0)
            lr = final_lr + 0.5 * (mid_lr - final_lr) * (1.0 + math.cos(math.pi * progress))

        return lr / base_lr

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
        last_epoch=last_epoch,
    )


def zero_center_gp(d_out, x_in):
    d_sum = d_out.sum()
    grad = torch.autograd.grad(
        outputs=d_sum,
        inputs=x_in,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.reshape(grad.size(0), -1)
    return (grad.pow(2).sum(dim=1)).mean()


def rpgan_losses(pred_real, pred_fake, all_pairs=True):
    pred_real = pred_real.reshape(pred_real.size(0), -1).mean(dim=1)
    pred_fake = pred_fake.reshape(pred_fake.size(0), -1).mean(dim=1)

    if all_pairs:
        diff = pred_real[:, None] - pred_fake[None, :]
    else:
        diff = pred_real - pred_fake

    loss_d_adv = F.softplus(-diff).mean()
    loss_g_adv = F.softplus(diff).mean()
    return loss_d_adv, loss_g_adv


def crop_image_by_part(image, part):
    h, w = image.shape[2:]
    h2, w2 = h // 2, w // 2

    if part == 0:
        return image[:, :, :h2, :w2]
    if part == 1:
        return image[:, :, :h2, w2:]
    if part == 2:
        return image[:, :, h2:, :w2]
    if part == 3:
        return image[:, :, h2:, w2:]
    raise ValueError(f"Invalid part: {part}")


def resize_to(x, ref):
    return F.interpolate(
        x,
        size=ref.shape[2:],
        mode="bilinear",
        align_corners=False,
    )


def build_lpips(device):
    if hasattr(lpips, "PerceptualLoss"):
        percept = lpips.PerceptualLoss(
            model="net-lin",
            net="vgg",
            use_gpu=(device.type == "cuda"),
        )
    else:
        percept = lpips.LPIPS(net="vgg")

    percept = percept.to(device)
    percept.eval()
    for p in percept.parameters():
        p.requires_grad_(False)
    return percept


def percept_loss(percept, x, y):
    return percept(x, y).mean()


# -----------------------------
# EMA
# -----------------------------
class EMA:
    def __init__(self, model: nn.Module, decay=0.999, device=None):
        self.model = model
        self.decay = decay
        self.device = device
        self.shadow = {}
        self.backup = None
        self.register()

    def register(self):
        self.shadow = {}
        for k, v in self.model.state_dict().items():
            t = v.detach().clone()
            if self.device is not None:
                t = t.to(self.device)
            self.shadow[k] = t

    @torch.no_grad()
    def update(self):
        current = self.model.state_dict()
        for k, v in current.items():
            src = v.detach()
            dst = self.shadow[k]
            src = src.to(dst.device)

            if torch.is_floating_point(src):
                dst.mul_(self.decay).add_(src, alpha=1 - self.decay)
            else:
                dst.copy_(src)

    def apply_to(self):
        self.backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.shadow, strict=True)

    def restore(self):
        if self.backup is not None:
            self.model.load_state_dict(self.backup, strict=True)
            self.backup = None

    def load_shadow(self, shadow_state):
        self.shadow = {}
        for k, v in shadow_state.items():
            t = v.detach().clone()
            if self.device is not None:
                t = t.to(self.device)
            self.shadow[k] = t


# -----------------------------
# Logging utils
# -----------------------------
def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def ensure_initial_lr(optimizer):
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])


def append_metrics_row(csv_path, row):
    file_exists = os.path.exists(csv_path)
    fieldnames = [
        "iteration",
        "loss_d_adv",
        "loss_g_adv",
        "loss_rec",
        "r1",
        "r2",
        "lr_g",
        "lr_d",
    ]

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_metric_history(csv_path):
    history = {
        "iteration": [],
        "loss_d_adv": [],
        "loss_g_adv": [],
        "loss_rec": [],
        "r1": [],
        "r2": [],
        "lr_g": [],
        "lr_d": [],
    }

    if not os.path.exists(csv_path):
        return history

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            history["iteration"].append(int(row["iteration"]))
            history["loss_d_adv"].append(float(row["loss_d_adv"]))
            history["loss_g_adv"].append(float(row["loss_g_adv"]))
            history["loss_rec"].append(float(row["loss_rec"]))
            history["r1"].append(float(row["r1"]))
            history["r2"].append(float(row["r2"]))
            history["lr_g"].append(float(row["lr_g"]))
            history["lr_d"].append(float(row["lr_d"]))

    return history


def update_history(history, row):
    history["iteration"].append(int(row["iteration"]))
    history["loss_d_adv"].append(float(row["loss_d_adv"]))
    history["loss_g_adv"].append(float(row["loss_g_adv"]))
    history["loss_rec"].append(float(row["loss_rec"]))
    history["r1"].append(float(row["r1"]))
    history["r2"].append(float(row["r2"]))
    history["lr_g"].append(float(row["lr_g"]))
    history["lr_d"].append(float(row["lr_d"]))


def plot_metrics(history, save_path):
    if len(history["iteration"]) == 0:
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(history["iteration"], history["loss_d_adv"], label="D_adv")
    axes[0].plot(history["iteration"], history["loss_g_adv"], label="G_adv")
    axes[0].legend()
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Adv Loss")
    axes[0].set_title("GAN Adversarial Loss")

    axes[1].plot(history["iteration"], history["loss_rec"], label="Rec")
    axes[1].plot(history["iteration"], history["r1"], label="R1")
    axes[1].plot(history["iteration"], history["r2"], label="R2")
    axes[1].legend()
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Aux Loss")
    axes[1].set_title("Reconstruction / Regularization")

    axes[2].plot(history["iteration"], history["lr_g"], label="LR_G")
    axes[2].plot(history["iteration"], history["lr_d"], label="LR_D")
    axes[2].legend()
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("LR")
    axes[2].set_title("Learning Rate")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# -----------------------------
# Train
# -----------------------------
def train(args):
    total_iterations = args.iter
    batch_size = args.batch_size
    im_size = args.im_size
    checkpoint = args.ckpt

    ndf = 64
    ngf = 64
    nz = 100
    latent_dim = 100

    lr = args.lr
    beta1 = 0.5
    workers = args.workers
    save_interval = args.save_interval

    saved_model_folder, saved_image_folder = get_dir(args)
    log_csv_path = os.path.join(saved_model_folder, "metrics.csv")
    loss_fig_path = os.path.join(saved_image_folder, "loss_curve.png")

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.cuda}" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
    print("Device:", device)

    # 按你的要求：数据路径保持写死
    img_path = "/home/ai//code/data/afhq/stargan-v2/data/train/cat"

    transform = transforms.Compose(
        [
            transforms.Resize((im_size, im_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    dataset = PoemImageDataset(img_dir=img_path, transform=transform)

    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        sampler=InfiniteSamplerWrapper(dataset),
        num_workers=workers,
        pin_memory=use_cuda,
    )
    if workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 2

    dataloader = DataLoader(**dataloader_kwargs)
    dataloader = iter(dataloader)

    fixed_z = torch.randn(batch_size, latent_dim, device=device)
    percept = build_lpips(device)

    netG = Generator(ngf=ngf, nz=nz, im_size=im_size).apply(weights_init).to(device)
    netD = Discriminator(ndf=ndf, im_size=im_size).apply(weights_init).to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

    ema = EMA(netG, decay=0.999, device=None)

    current_iteration = args.start_iter
    ckpt = None

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location="cpu")

        netG.load_state_dict(ckpt["g"], strict=True)
        netD.load_state_dict(ckpt["d"], strict=True)
        optimizerG.load_state_dict(ckpt["opt_g"])
        optimizerD.load_state_dict(ckpt["opt_d"])
        optimizer_to_device(optimizerG, device)
        optimizer_to_device(optimizerD, device)

        if "g_ema" in ckpt:
            ema.load_shadow(ckpt["g_ema"])
        else:
            ema.register()

        current_iteration = ckpt.get("iter", -1) + 1

        print(f"Resumed from: {checkpoint}")
        print(f"Start iteration: {current_iteration}")

    ensure_initial_lr(optimizerG)
    ensure_initial_lr(optimizerD)

    if ckpt is not None and "sch_g" in ckpt and "sch_d" in ckpt:
        scheduler_G = get_three_stage_scheduler(
            optimizerG,
            total_iterations,
            warmup_end=2000,
            decay_end=20000,
            peak_lr=5e-4,
            mid_lr=2e-4,
            min_lr_ratio=0.01,
            last_epoch=-1,
        )
        scheduler_D = get_three_stage_scheduler(
            optimizerD,
            total_iterations,
            warmup_end=2000,
            decay_end=20000,
            peak_lr=5e-4,
            mid_lr=2e-4,
            min_lr_ratio=0.01,
            last_epoch=-1,
        )
        scheduler_G.load_state_dict(ckpt["sch_g"])
        scheduler_D.load_state_dict(ckpt["sch_d"])
    else:
        scheduler_G = get_three_stage_scheduler(
            optimizerG,
            total_iterations,
            warmup_end=2000,
            decay_end=20000,
            peak_lr=5e-4,
            mid_lr=2e-4,
            min_lr_ratio=0.01,
            last_epoch=current_iteration - 1,
        )
        scheduler_D = get_three_stage_scheduler(
            optimizerD,
            total_iterations,
            warmup_end=2000,
            decay_end=20000,
            peak_lr=5e-4,
            mid_lr=2e-4,
            min_lr_ratio=0.01,
            last_epoch=current_iteration - 1,
        )

    history = load_metric_history(log_csv_path)

    for iteration in tqdm(range(current_iteration, total_iterations + 1)):
        # -----------------------------
        # 1) Real batch
        # -----------------------------
        real_image = next(dataloader).to(device, non_blocking=True)
        bsz = real_image.size(0)

        # -----------------------------
        # 2) D step
        # -----------------------------
        optimizerD.zero_grad(set_to_none=True)

        with torch.no_grad():
            z_d = torch.randn(bsz, latent_dim, device=device)
            fake_images_d = netG(z_d)

        real_aug = DiffAugment(real_image, policy=policy)
        fake_aug_d = [DiffAugment(fake, policy=policy) for fake in fake_images_d]

        pred_real = netD.forward_logits(real_aug)
        pred_fake = netD.forward_logits([fi.detach() for fi in fake_aug_d])

        loss_d_adv, _ = rpgan_losses(pred_real, pred_fake, all_pairs=True)

        if iteration < 40000:
            part = random.randint(0, 3)
            rec_all, rec_small, rec_part = netD.forward_recon(real_image, part=part)

            target_all = resize_to(real_image, rec_all)
            target_small = resize_to(real_image, rec_small)
            target_part = resize_to(crop_image_by_part(real_image, part), rec_part)

            loss_rec_all = percept_loss(percept, rec_all, target_all)
            loss_rec_small = percept_loss(percept, rec_small, target_small)
            loss_rec_part = percept_loss(percept, rec_part, target_part)
            loss_rec = loss_rec_all + loss_rec_small + loss_rec_part

            loss_D = loss_d_adv + (1 - iteration / 40000) * loss_rec
        else:
            loss_rec = torch.tensor(0.0, device=device)
            loss_D = loss_d_adv

        real_r1 = real_aug.detach().requires_grad_(True)
        d_real_for_r1 = netD.forward_logits(real_r1)
        r1 = zero_center_gp(d_real_for_r1, real_r1)

        fake0 = fake_aug_d[0].detach().requires_grad_(True)
        d_fake_for_r2 = netD.forward_logits(fake0)
        r2 = zero_center_gp(d_fake_for_r2, fake0)

        loss_D = loss_D + 0.1 * r1 + 0.1 * r2
        loss_D.backward()
        optimizerD.step()

        # -----------------------------
        # 3) G step
        # -----------------------------
        optimizerG.zero_grad(set_to_none=True)

        z_g = torch.randn(bsz, latent_dim, device=device)
        fake_images_g = netG(z_g)
        fake_aug_g = [DiffAugment(fake, policy=policy) for fake in fake_images_g]

        with torch.no_grad():
            pred_real_g = netD.forward_logits(real_aug)

        pred_fake_g = netD.forward_logits(fake_aug_g)
        _, loss_G = rpgan_losses(pred_real_g, pred_fake_g, all_pairs=True)

        loss_G.backward()
        optimizerG.step()

        ema.update()
        scheduler_G.step()
        scheduler_D.step()

        # -----------------------------
        # 4) Logging
        # -----------------------------
        row = {
            "iteration": iteration,
            "loss_d_adv": float(loss_d_adv.item()),
            "loss_g_adv": float(loss_G.item()),
            "loss_rec": float(loss_rec.item()),
            "r1": float(r1.item()),
            "r2": float(r2.item()),
            "lr_g": float(optimizerG.param_groups[0]["lr"]),
            "lr_d": float(optimizerD.param_groups[0]["lr"]),
        }

        append_metrics_row(log_csv_path, row)
        update_history(history, row)

        if iteration % 100 == 0:
            print(
                f"[{iteration}/{total_iterations}] "
                f"D_adv: {loss_d_adv.item():.5f} | "
                f"G_adv: {loss_G.item():.5f} | "
                f"Rec: {loss_rec.item():.5f} | "
                f"R1: {r1.item():.5f} | "
                f"R2: {r2.item():.5f} | "
                f"LR_G: {optimizerG.param_groups[0]['lr']:.7f} | "
                f"LR_D: {optimizerD.param_groups[0]['lr']:.7f}"
            )

        if iteration % (save_interval * 10) == 0 or iteration == total_iterations:
            plot_metrics(history, loss_fig_path)
            print("Loss curve saved to:", loss_fig_path)

        # -----------------------------
        # 5) Save images
        # -----------------------------
        if iteration % (save_interval * 10) == 0 or iteration == total_iterations:
            with torch.no_grad():
                ema.apply_to()
                fixed_fake = netG(fixed_z)[0]
                ema.restore()

                img_grid = ((fixed_fake + 1.0) / 2.0).clamp(0.0, 1.0)
                save_image(
                    img_grid,
                    f"{saved_image_folder}/iter_{iteration}.jpg",
                    nrow=min(8, img_grid.size(0)),
                )

        # -----------------------------
        # 6) Save checkpoints
        # -----------------------------
        if iteration % (save_interval * 50) == 0 or iteration == total_iterations:
            torch.save(
                {
                    "iter": iteration,
                    "g": netG.state_dict(),
                    "g_ema": {k: v.detach().cpu() for k, v in ema.shadow.items()},
                    "d": netD.state_dict(),
                    "opt_g": optimizerG.state_dict(),
                    "opt_d": optimizerD.state_dict(),
                    "sch_g": scheduler_G.state_dict(),
                    "sch_d": scheduler_D.state_dict(),
                },
                os.path.join(saved_model_folder, f"all_{iteration}.pth"),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="region gan")

    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="path of resource dataset, should be a folder that has one or many sub image folders inside",
    )
    parser.add_argument("--output_path", type=str, default="./", help="Output path for the train results")
    parser.add_argument("--cuda", type=int, default=0, help="index of gpu to use")
    parser.add_argument("--name", type=str, default="KKKK", help="experiment name")
    parser.add_argument("--iter", type=int, default=200000, help="number of iterations")
    parser.add_argument("--start_iter", type=int, default=0, help="the iteration to start training")
    parser.add_argument("--batch_size", type=int, default=64, help="mini batch number of images")
    parser.add_argument("--im_size", type=int, default=256, help="image resolution")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint weight path if have one")
    parser.add_argument("--workers", type=int, default=2, help="number of workers for dataloader")
    parser.add_argument("--save_interval", type=int, default=100, help="number of iterations to save model")
    parser.add_argument("--lr", type=float, default=0.0005, help="learn")

    args = parser.parse_args()
    print(args)

    train(args)