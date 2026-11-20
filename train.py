"""Train TinyCNN on the generated shape images."""
import argparse

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from model import TinyCNN

MEAN, STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_split(folder, tf, device):
    """Load a whole ImageFolder split into memory as (images, labels) tensors.

    The dataset is small (~100 MB), so this is much faster than a DataLoader.
    """
    ds = datasets.ImageFolder(folder, tf)
    x = torch.stack([img for img, _ in ds]).to(device)
    y = torch.tensor(ds.targets, device=device)
    return x, y, ds.classes


def augment(x):
    """Random flips and 90-degree rotations, applied per batch on-device."""
    if torch.rand(1).item() < 0.5:
        x = x.flip(3)
    return torch.rot90(x, int(torch.randint(4, (1,))), dims=(2, 3))


def evaluate(model, loader, device, n_classes):
    model.eval()
    confusion = torch.zeros(n_classes, n_classes, dtype=torch.long)
    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(1).cpu()
            for t, p in zip(y.cpu(), preds):
                confusion[t, p] += 1
    acc = confusion.diag().sum().item() / confusion.sum().item()
    return acc, confusion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--out", default="shape_cnn.pt")
    args = p.parse_args()

    device = get_device()
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_x, train_y, classes = load_split(f"{args.data}/train", tf, device)
    val_x, val_y, _ = load_split(f"{args.data}/val", tf, device)
    train_dl = list(zip(train_x.split(args.batch_size), train_y.split(args.batch_size)))
    val_dl = list(zip(val_x.split(256), val_y.split(256)))
    model = TinyCNN(len(classes)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  classes={classes}  params={n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = correct = seen = 0
        perm = torch.randperm(len(train_y), device=device)
        for idx in perm.split(args.batch_size):
            x, y = augment(train_x[idx]), train_y[idx]
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        sched.step()

        val_acc, _ = evaluate(model, val_dl, device, len(classes))
        print(f"epoch {epoch:2d}  loss {total_loss / seen:.4f}  "
              f"train_acc {correct / seen:.3f}  val_acc {val_acc:.3f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({"state_dict": model.state_dict(), "classes": classes}, args.out)

    model.load_state_dict(torch.load(args.out)["state_dict"])
    val_acc, confusion = evaluate(model, val_dl, device, len(classes))
    print(f"\nbest val_acc {val_acc:.3f}  saved to {args.out}")
    print("confusion matrix (rows = true, cols = predicted):")
    w = max(len(c) for c in classes)
    print(" " * (w + 2) + " ".join(f"{c[:5]:>5}" for c in classes))
    for name, row in zip(classes, confusion):
        print(f"{name:>{w}}  " + " ".join(f"{v:5d}" for v in row.tolist()))


if __name__ == "__main__":
    main()
