"""U-Net for CD-SEM images: pixel classes space / line / particle / bridge / break.

One network does both jobs:
  * metrology  - the line probability map gives sub-pixel edges row by row -> CD and LER
  * inspection - defect pixels give the defect type AND its location
"""
import time, json, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
N_CLS = 5


def block(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                         nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, base=16, n_cls=N_CLS):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.e1, self.e2, self.e3, self.e4 = block(1, c[0]), block(c[0], c[1]), block(c[1], c[2]), block(c[2], c[3])
        self.mid = block(c[3], c[3])
        self.u4, self.d4 = nn.ConvTranspose2d(c[3], c[3], 2, 2), block(c[3] * 2, c[2])
        self.u3, self.d3 = nn.ConvTranspose2d(c[2], c[2], 2, 2), block(c[2] * 2, c[1])
        self.u2, self.d2 = nn.ConvTranspose2d(c[1], c[1], 2, 2), block(c[1] * 2, c[0])
        self.u1, self.d1 = nn.ConvTranspose2d(c[0], c[0], 2, 2), block(c[0] * 2, c[0])
        self.out = nn.Conv2d(c[0], n_cls, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(F.max_pool2d(e1, 2)); e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2)); m = self.mid(F.max_pool2d(e4, 2))
        d = self.d4(torch.cat([self.u4(m), e4], 1)); d = self.d3(torch.cat([self.u3(d), e3], 1))
        d = self.d2(torch.cat([self.u2(d), e2], 1)); d = self.d1(torch.cat([self.u1(d), e1], 1))
        return self.out(d)


def to_tensor(imgs):
    x = torch.from_numpy(imgs).float().div(255.0).unsqueeze(1)
    return (x - 0.35) / 0.2


def augment(x, y):
    if np.random.rand() < 0.5: x, y = x.flip(-1), y.flip(-1)
    if np.random.rand() < 0.5: x, y = x.flip(-2), y.flip(-2)
    g = torch.empty(x.size(0), 1, 1, 1).uniform_(0.8, 1.25); b = torch.empty(x.size(0), 1, 1, 1).uniform_(-0.3, 0.3)
    return x * g + b, y


def loss_fn(logits, y, w):
    ce = F.cross_entropy(logits, y, weight=w)
    p = logits.softmax(1)
    oh = F.one_hot(y, N_CLS).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3)); den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    dice = 1 - (2 * inter + 1) / (den + 1)
    return ce + dice[1:].mean()                 # soft Dice on line + the three defect classes


@torch.no_grad()
def predict(model, imgs, bs=16):
    model.eval(); out = []
    for i in range(0, len(imgs), bs):
        out.append(model(to_tensor(imgs[i:i + bs])).softmax(1).numpy())
    return np.concatenate(out)


def train(epochs=14, bs=8, lr=3e-3, out="models/sem_unet.pt", init=None):
    tr = np.load("data/sem_train.npz"); va = np.load("data/sem_val.npz")
    Xtr, Ytr = to_tensor(tr["imgs"]), torch.from_numpy(tr["labs"]).long()
    Xva, Yva = va["imgs"], va["labs"]
    freq = np.bincount(tr["labs"].ravel(), minlength=N_CLS) / tr["labs"].size
    w = torch.tensor(np.clip((1 / np.sqrt(freq + 1e-6)) / (1 / np.sqrt(freq[0])), 1, 30), dtype=torch.float32)
    model = UNet()
    if init:                                     # continue training from an earlier checkpoint
        model.load_state_dict(torch.load(init, map_location="cpu"))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(Xtr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    log = {"class_weights": w.tolist(), "epochs": []}
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr)); t0 = time.time(); tot = 0
        for i in range(0, len(Xtr) - bs + 1, bs):
            idx = perm[i:i + bs]; x, y = augment(Xtr[idx], Ytr[idx])
            loss = loss_fn(model(x), y, w)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        P = predict(model, Xva).argmax(1)
        iou = [((P == c) & (Yva == c)).sum() / max(((P == c) | (Yva == c)).sum(), 1) for c in range(N_CLS)]
        rec = {"epoch": ep + 1, "loss": tot / (len(Xtr) // bs), "val_iou": [round(float(v), 3) for v in iou],
               "sec": round(time.time() - t0)}
        log["epochs"].append(rec); print(rec, flush=True)
        torch.save(model.state_dict(), out)
    json.dump(log, open("models/sem_unet_trainlog.json", "w"), indent=1)


if __name__ == "__main__":
    train(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 14, init=sys.argv[2] if len(sys.argv) > 2 else None)
