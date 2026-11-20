import os
import glob
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from wafer_metrology import DefectUNet
# ==========================================
# 1. HARDWARE SETUP
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Training on: {device}")
if device.type == "cuda":
    print(f"[*] Device: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

IMG_SIZE = 768

# ==========================================
# 2. RAM-CACHED DATASET (Zero disk I/O per epoch)
# ==========================================
def load_dataset_to_ram(img_files, mask_files, img_size=IMG_SIZE):
    n = len(img_files)
    imgs = torch.empty((n, 1, img_size, img_size), dtype=torch.uint8)
    masks = torch.empty((n, 1, img_size, img_size), dtype=torch.uint8)

    t0 = time.time()
    for i, (ip, mp) in enumerate(zip(img_files, mask_files)):
        im = Image.open(ip).convert("L").resize((img_size, img_size), Image.BILINEAR)
        mk = Image.open(mp).convert("L").resize((img_size, img_size), Image.BILINEAR)
        imgs[i, 0] = torch.from_numpy(np.array(im, dtype=np.uint8))
        masks[i, 0] = torch.from_numpy((np.array(mk, dtype=np.uint8) > 128).astype(np.uint8))

    print(f"[*] Preloaded {n} pairs into RAM ({imgs.nbytes*2/1e6:.0f} MB) in {time.time()-t0:.1f}s")
    return imgs, masks

class CachedWaferDataset(Dataset):
    def __init__(self, imgs, masks, augment=False):
        self.imgs = imgs
        self.masks = masks
        self.augment = augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx].float() / 255.0
        mask = self.masks[idx].float()

        if self.augment:
            if random.random() > 0.5:
                img = torch.flip(img, dims=[2])
                mask = torch.flip(mask, dims=[2])
            if random.random() > 0.5:
                img = torch.flip(img, dims=[1])
                mask = torch.flip(mask, dims=[1])
            k = random.randint(0, 3)
            if k > 0:
                img = torch.rot90(img, k, dims=[1, 2])
                mask = torch.rot90(mask, k, dims=[1, 2])

        return img, mask

# ==========================================
# 3. U-NET (Low VRAM, 4-stage)
# ==========================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        )
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], dim=1))

class DefectUNet(nn.Module):
    def __init__(self, base_ch=24):
        super().__init__()
        c = base_ch
        self.enc1 = ConvBlock(1, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(c * 8, c * 16)

        self.dec4 = UpBlock(c * 16, c * 8)
        self.dec3 = UpBlock(c * 8, c * 4)
        self.dec2 = UpBlock(c * 4, c * 2)
        self.dec1 = UpBlock(c * 2, c)
        self.final = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.final(d1)

# ==========================================
# 4. LOSS & METRIC
# ==========================================
class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = logits.float()
        targets = targets.float()
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
        dice_loss = 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)
        return bce_loss + dice_loss

@torch.no_grad()
def compute_dice(logits, targets, eps=1e-6):
    preds = (torch.sigmoid(logits.float()) > 0.5).float()
    inter = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    return ((2.0 * inter + eps) / (union + eps)).item()

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def train():
    img_files = sorted(glob.glob(os.path.join("dataset", "images", "*.png")))
    mask_files = sorted(glob.glob(os.path.join("dataset", "masks", "*.png")))
    assert len(img_files) > 0 and len(img_files) == len(mask_files), "Dataset missing or mismatched!"

    imgs, masks = load_dataset_to_ram(img_files, mask_files, IMG_SIZE)

    indices = list(range(len(imgs)))
    random.Random(42).shuffle(indices)
    val_size = max(1, int(0.15 * len(indices)))
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    train_data = CachedWaferDataset(imgs[train_idx], masks[train_idx], augment=True)
    val_data = CachedWaferDataset(imgs[val_idx], masks[val_idx], augment=False)

    # num_workers=0 is fastest when data is already cached in RAM
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False, pin_memory=(device.type == "cuda"))

    model = DefectUNet(base_ch=16).to(device)
    criterion = BCEDiceLoss().to(device)

    EPOCHS = 70
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    best_dice = 0.0
    print(f"\n[*] Training ({EPOCHS} epochs, {len(train_data)} train / {len(val_data)} val)...")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * x.size(0)

        scheduler.step()

        model.eval()
        val_loss, val_dice = 0.0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    out = model(x)
                    loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                val_dice += compute_dice(out, y) * x.size(0)

        train_loss /= len(train_data)
        val_loss /= len(val_data)
        val_dice /= len(val_data)
        vram_gb = torch.cuda.max_memory_allocated() / 1e9 if use_amp else 0.0

        print(
            f"Epoch [{epoch:02d}/{EPOCHS:02d}] ({time.time()-t0:.1f}s | {vram_gb:.2f} GB) "
            f"- Train: {train_loss:.4f} | Val: {val_loss:.4f} | Val Dice: {val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "defect_unet.pth")

    print(f"\n[+] Done! Best Val Dice: {best_dice:.4f} saved to defect_unet.pth")

if __name__ == "__main__":
    train()