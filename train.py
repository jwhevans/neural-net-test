"""Train TinyCNN on the generated shape images.

Anything that has to match predict.py lives in config.py; anything that is a
run choice is a CLI flag whose default also lives in config.py. set_seed() pins
the Python, NumPy and PyTorch RNGs, so the same command reproduces the same run
on the same device.

Split discipline: `train` is what the weights see, `val` is what model
selection, threshold tuning and the diagnostics are allowed to look at, and
`test` is read once, after the weights are already frozen, and printed as-is.
"""
import argparse
import copy
import os
import random
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

from config import (ARCH, AUG_BRIGHTNESS, AUG_CONTRAST, BALANCED_SAMPLING,
                    BATCH_SIZE, CAP_TRAIN_IMAGES, CKPT_VERSION, DATA_DIR, EPOCHS,
                    FLOOR_TRAIN_IMAGES, LABEL_SMOOTHING, LR, MEAN, MODEL_PATH,
                    SEED, SIZE, STD, TEST_SPLIT, THRESHOLD_ACC, TRAIN_SPLIT,
                    VAL_SPLIT, WEIGHT_DECAY, WORKERS)
from model import TinyCNN


def set_seed(seed):
    """Pin every RNG that can move a run.

    DataLoader workers derive their seeds from the torch RNG, so seeding it here
    (before the loaders are iterated) fixes the augmentation draws too.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Rot90:
    """Random 0/90/180/270 degree rotation of a CHW tensor.

    Multiples of 90 degrees cannot clip, which is what makes rotation safe for a
    shape that already fills most of a 64x64 frame: an arbitrary angle drags
    pixels outside the frame and leaves a mangled shape carrying a label the
    image no longer supports.
    """

    def __call__(self, x):
        k = int(torch.randint(4, (1,)).item())
        return torch.rot90(x, k, dims=(1, 2)) if k else x


def eval_tf():
    """The preprocessing every evaluation path must share with the model."""
    return transforms.Compose([
        transforms.Resize((SIZE, SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def train_tf(augment_on):
    """Training-only augmentation.

    Flips and 90-degree rotations are lossless here - a flipped or rotated shape
    is still the same shape - and colour jitter stops the net keying on the
    exact foreground/background pairs the generator happens to draw. Nothing
    crops, translates or rotates by an arbitrary angle, because generate_data.py
    keeps every shape inside the frame and such a transform would train on
    clipped shapes under labels those shapes do not support.
    """
    steps = [transforms.Resize((SIZE, SIZE), antialias=True)]
    if augment_on:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=AUG_BRIGHTNESS, contrast=AUG_CONTRAST),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if augment_on:
        steps.append(Rot90())               # works on the tensor, so after ToTensor
    return transforms.Compose(steps)


def targets_of(ds):
    """Labels of an ImageFolder or of a Subset of one."""
    if isinstance(ds, Subset):
        return [ds.dataset.targets[i] for i in ds.indices]
    return list(ds.targets)


def rebalanced(ds, cap, floor):
    """Optional floor-and-cap on the training split (0 turns either off).

    `cap` throws away the surplus of every over-represented class; `floor` then
    repeats the smallest classes up to that size. Both are off unless asked for:
    a deliberately imbalanced fine-tune set is usually the thing being measured,
    and quietly resampling it hides the effect the run is meant to show.
    """
    targets = targets_of(ds)
    counts = Counter(targets)
    by_class = {c: [i for i, t in enumerate(targets) if t == c] for c in counts}
    keep = []
    for c, idx in sorted(by_class.items()):
        if cap and len(idx) > cap:
            idx = idx[:cap]
        if floor and len(idx) < floor:
            idx = (idx * (floor // len(idx) + 1))[:floor]
        keep.extend(idx)
    return Subset(ds, sorted(keep))


def sampler_for(ds, n_classes, seed):
    """Each sample's weight is 1 / (its class count * classes): equal class shares."""
    targets = targets_of(ds)
    counts = Counter(targets)
    weights = [1.0 / (counts[t] * n_classes) for t in targets]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True,
                                 generator=torch.Generator().manual_seed(seed))


def make_loaders(args, n_classes=None):
    """Build the three split loaders and check the label maps really agree."""
    eval_only = eval_tf()
    splits = {}
    for name, tf in ((TRAIN_SPLIT, train_tf(not args.no_augmentation)),
                     (VAL_SPLIT, eval_only), (TEST_SPLIT, eval_only)):
        folder = os.path.join(args.data, name)
        if not os.path.isdir(folder):
            sys.exit(f"missing split folder {folder} - run generate_data.py first")
        splits[name] = datasets.ImageFolder(folder, tf)

    classes = splits[TRAIN_SPLIT].classes
    for name in (VAL_SPLIT, TEST_SPLIT):
        if splits[name].classes != classes:
            sys.exit(f"class folders differ: {TRAIN_SPLIT}={classes} "
                     f"{name}={splits[name].classes}")
    K = len(classes)
    if n_classes and n_classes != K:
        sys.exit(f"checkpoint head outputs {n_classes} classes, data has {K}")

    train_ds = splits[TRAIN_SPLIT]
    if args.cap_train_images or args.floor_train_images:
        train_ds = rebalanced(train_ds, args.cap_train_images, args.floor_train_images)
    sampler = sampler_for(train_ds, K, args.seed) if args.balanced_sampling else None
    pin = args.device_obj.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=sampler is None, sampler=sampler,
                              num_workers=args.workers, pin_memory=pin,
                              persistent_workers=args.workers > 0)
    eval_loaders = {name: DataLoader(splits[name], batch_size=256, shuffle=False,
                                     num_workers=args.workers, pin_memory=pin,
                                     persistent_workers=args.workers > 0)
                    for name in (VAL_SPLIT, TEST_SPLIT)}
    return train_ds, train_loader, eval_loaders, classes


@torch.no_grad()
def collect(model, loader, device):
    """One forward pass per image -> (probs, preds, targets), all on CPU."""
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        probs.append(model(x.to(device)).softmax(1).cpu())
        targets.append(y)
    probs = torch.cat(probs)
    return probs, probs.argmax(1), torch.cat(targets)


def scores(preds, targets, n_classes):
    """Accuracy plus macro-F1 (confusion rows = true class, cols = predicted).

    Accuracy alone is the wrong headline for an imbalanced fine-tune set: if 90%
    of the images are one class, always guessing that class scores 0.90. Macro-F1
    averages the per-class F1, so a class nobody ever predicts drags the number
    down to what it actually is.
    """
    conf = torch.bincount(targets * n_classes + preds, minlength=n_classes ** 2)
    conf = conf.view(n_classes, n_classes)
    actual, predicted = conf.sum(1), conf.sum(0)
    total = conf.sum().item()
    acc = conf.diag().sum().item() / total if total else 0.0
    f1s = []
    for c in range(n_classes):
        tp = conf[c, c].item()
        prec = tp / predicted[c].item() if predicted[c].item() else 0.0
        rec = tp / actual[c].item() if actual[c].item() else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {"acc": acc, "macro_f1": sum(f1s) / n_classes, "per_class_f1": f1s,
            "support": actual.tolist(), "confusion": conf.tolist()}


def report(name, m, classes):
    per = " ".join(f"{c[:6]}={v:.3f}" for c, v in zip(classes, m["per_class_f1"]))
    print(f"{name:>5}: acc {m['acc']:.4f}  macro-F1 {m['macro_f1']:.4f}  "
          f"n={sum(m['support'])}  [{per}]")


def matrix(rows, classes, fmt="{:+.3f}"):
    width = max(len(fmt.format(v)) for row in rows for v in row)
    width = max(width, 7)
    print(" " * 11 + " ".join(f"{c[:width]:>{width}}" for c in classes))
    for name, row in zip(classes, rows):
        print(f"{name[:9]:>9}  " + " ".join(f"{fmt.format(v):>{width}}" for v in row))


def confusion_report(m, classes):
    print("confusion (rows = true, cols = predicted)")
    matrix(m["confusion"], classes, fmt="{:d}")


def threshold_sweep(probs, preds, targets, classes, target_acc):
    """Pick a per-class acceptance threshold on val.

    A prediction of class c is accepted when its confidence is >= t[c]. For each
    class the sweep picks the lowest t whose accepted predictions of c are at
    least `target_acc` correct - the lowest bar keeps the most images. Every
    class gets its own bar because one class saturates at 0.999 confidence while
    another is genuinely harder. Tuned on val only: tune it on test and the test
    number stops being an estimate of anything.
    """
    conf = probs.gather(1, preds[:, None]).squeeze(1)
    print(f"\nper-class acceptance threshold on {VAL_SPLIT} (lowest bar whose "
          f"accepted predictions reach {target_acc:.2f} accuracy)")
    picked = []
    for c, name in enumerate(classes):
        pred_c = preds == c
        if int(pred_c.sum()) == 0:
            print(f"{name[:9]:>9}  never predicted on {VAL_SPLIT} - threshold 1.00")
            picked.append(1.0)
            continue
        threshold = None
        for t in torch.linspace(0.0, 1.0, 101).tolist():
            acc_mask = pred_c & (conf >= t)
            n = int(acc_mask.sum())
            if n == 0:
                break
            if int((acc_mask & (targets == c)).sum()) / n >= target_acc:
                threshold = t
                break
        if threshold is None:
            print(f"{name[:9]:>9}  no bar reaches {target_acc:.2f} - threshold 1.00 "
                  f"(rejects everything)")
            picked.append(1.0)
            continue
        acc_mask = pred_c & (conf >= threshold)
        n = int(acc_mask.sum())
        print(f"{name[:9]:>9}  t={threshold:.2f}  accepts {n}/{int(pred_c.sum())} "
              f"predictions  accepted_acc {int((acc_mask & (targets == c)).sum()) / n:.3f}")
        picked.append(threshold)
    return picked


def apply_thresholds(name, probs, preds, targets, classes, thresholds):
    """Coverage and accuracy of the accepted predictions under fixed thresholds."""
    conf = probs.gather(1, preds[:, None]).squeeze(1)
    bar = torch.tensor(thresholds)[preds]
    accepted = conf >= bar
    n = int(accepted.sum())
    acc = int((accepted & (preds == targets)).sum()) / n if n else 0.0
    print(f"{name:>5} with {VAL_SPLIT} thresholds: accepts {n}/{len(preds)} "
          f"({n / len(preds):.1%})  accepted_acc {acc:.4f}")


@torch.no_grad()
def prototype_report(model, loader, device, classes):
    """Where the model's features put one class close to another.

    Features are the pooled conv output (the input to the linear head). They
    come straight out of a ReLU, so every component is >= 0 and raw cosines
    between any two of them sit near 1 whatever the class. Subtracting the mean
    feature of the split first removes that shared offset; what is left is
    L2-normalised, so comparisons are cosines that can go negative and actually
    separate classes. A class prototype is the normalised mean of that class's
    unit feature vectors.

    The first table is how a true-class image scores against each prototype, the
    second is prototype-to-prototype, ranked: the closest pairs are the classes
    the features separate least, and the per-class F1 line is where that lands.
    """
    model.eval()
    feats, targets = [], []
    for x, y in loader:
        f = model.head[:2](model.features(x.to(device)))    # avg-pool + flatten
        feats.append(f.cpu())
        targets.append(y)
    feats = torch.cat(feats)
    feats = nn.functional.normalize(feats - feats.mean(0), dim=1)
    targets = torch.cat(targets)
    K = len(classes)

    present = [c for c in range(K) if int((targets == c).sum())]
    protos = torch.zeros(K, feats.shape[1])
    for c in present:
        protos[c] = feats[targets == c].mean(0)
    protos = nn.functional.normalize(protos, dim=1)

    print(f"\nimage-to-prototype cosine on {VAL_SPLIT}, averaged over each class's "
          f"own images ({feats.shape[1]}-d pooled features, mean-centred, "
          f"L2-normalised)")
    sims = feats @ protos.T
    rows = [sims[targets == c].mean(0).tolist() if c in present else [0.0] * K
            for c in range(K)]
    matrix(rows, classes)
    print("diagonal = how firmly each class holds its own direction")

    proto_sim = protos @ protos.T
    print("\nprototype-to-prototype cosine")
    matrix(proto_sim.tolist(), classes)
    pairs = sorted(((proto_sim[i, j].item(), classes[i], classes[j])
                    for i in present for j in present if i < j), reverse=True)
    print("closest prototype pairs: "
          + ", ".join(f"{a}/{b} {v:+.3f}" for v, a, b in pairs[:3]))

    sims[:, [c for c in range(K) if c not in present]] = -2.0
    top2 = sims.topk(2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    nearest_acc = (sims.argmax(1) == targets).float().mean().item()
    print(f"\nnearest-prototype accuracy {nearest_acc:.4f}")
    print(f"images whose two closest prototypes are within 0.05: "
          f"{(margin < 0.05).float().mean():.3f}  (mean margin {margin.mean():.3f})")
    missing = [classes[c] for c in range(K) if c not in present]
    if missing:
        print(f"no images in this split for {missing} - no prototype for them")


def save_checkpoint(path, state_dict, classes, args, val_metrics=None, thresholds=None):
    """Weights plus everything predict.py needs to know they are being fed right.

    A checkpoint loaded with the wrong SIZE or MEAN/STD still runs and still
    prints confident numbers, so the `preprocess` block is what lets a loader
    refuse rather than guess. `classes` is the order ImageFolder assigned
    (alphabetical) and is the only thing that maps a logits vector back to a
    shape name.
    """
    payload = {
        "state_dict": state_dict,
        "classes": classes,
        "version": CKPT_VERSION,
        "arch": ARCH,
        "preprocess": {"size": SIZE, "mean": list(MEAN), "std": list(STD)},
        "train_config": {
            "seed": args.seed, "epochs": args.epochs, "lr": args.lr,
            "batch_size": args.batch_size, "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "augment": not args.no_augmentation,
            "balanced_sampling": args.balanced_sampling,
            "cap_train_images": args.cap_train_images,
            "floor_train_images": args.floor_train_images,
            "data_dir": args.data,
            "init": args.init,
        },
        "val_metrics": val_metrics,
        "thresholds": thresholds,
    }
    torch.save(payload, path)


def load_checkpoint(path):
    if not os.path.isfile(path):
        sys.exit(f"no checkpoint at {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def check_preprocess(ck, path):
    """A mismatched checkpoint still runs; it just means something else."""
    got = ck.get("preprocess")
    if got and got != {"size": SIZE, "mean": list(MEAN), "std": list(STD)}:
        sys.exit(f"{path} came from {got}, config.py says size={SIZE} "
                 f"mean={list(MEAN)} std={list(STD)} - retrain or match the config")


def train(args):
    """Train on `train`, select on `val`, and return the weights that won."""
    device = args.device_obj
    set_seed(args.seed)                       # before the loaders: shuffling and augmentation draw from this

    init = None
    if args.init:
        init = load_checkpoint(args.init)
        check_preprocess(init, args.init)
    train_set, train_loader, eval_loaders, classes = make_loaders(
        args, n_classes=len(init["classes"]) if init else None)
    if init and init["classes"] != classes:
        sys.exit(f"{args.init} was trained on {init['classes']}, data has {classes}")

    model = TinyCNN(len(classes)).to(device)
    if init:
        model.load_state_dict(init["state_dict"])
    n_params = sum(p.numel() for p in model.parameters())
    counts = Counter(targets_of(train_set))
    dist = "  ".join(f"{classes[c]}:{counts.get(c, 0)}" for c in range(len(classes)))
    print(f"device={device}  arch={ARCH}  params={n_params:,}"
          + (f"  init={args.init}" if init else ""))
    print(f"images  train {len(train_set)}  val {len(eval_loaders[VAL_SPLIT].dataset)}  "
          f"test {len(eval_loaders[TEST_SPLIT].dataset)}")
    print(f"train distribution  {dist}")
    print(f"sampling  {'balanced (equal share per class)' if args.balanced_sampling else 'unweighted'}")
    print(f"augmentation  {'off' if args.no_augmentation else 'flip + colour jitter + rot90'}")
    print(f"epochs {args.epochs}  batch {args.batch_size}  lr {args.lr:g}  "
          f"weight_decay {args.weight_decay:g}  label_smoothing {args.label_smoothing:g}\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    best, best_state, best_epoch = {"macro_f1": -1.0}, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        seen = loss_sum = 0
        tr_preds, tr_targets = [], []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            seen += y.numel()
            loss_sum += loss.item() * y.numel()
            tr_preds.append(logits.argmax(1).cpu())
            tr_targets.append(y.cpu())
        sched.step()
        tr = scores(torch.cat(tr_preds), torch.cat(tr_targets), len(classes))
        _, preds, targets = collect(model, eval_loaders[VAL_SPLIT], device)
        m = scores(preds, targets, len(classes))
        improved = m["macro_f1"] > best["macro_f1"]
        print(f"epoch {epoch:2d}  loss {loss_sum / seen:.4f}  train_acc {tr['acc']:.3f}  "
              f"train_f1 {tr['macro_f1']:.3f}  val_acc {m['acc']:.4f}  "
              f"val_f1 {m['macro_f1']:.4f}{'  *' if improved else ''}")
        if improved:
            best, best_epoch = m, epoch
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

    model.load_state_dict(best_state)
    print(f"\nbest weights from epoch {best_epoch} (val macro-F1 {best['macro_f1']:.4f})")
    return model, classes, best, eval_loaders


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=DATA_DIR)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--workers", type=int, default=WORKERS)
    p.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    p.add_argument("--no-augmentation", action="store_true")
    p.add_argument("--balanced-sampling", action="store_true", default=BALANCED_SAMPLING)
    p.add_argument("--cap-train-images", type=int, default=CAP_TRAIN_IMAGES,
                   help="max training images per class (0 = off)")
    p.add_argument("--floor-train-images", type=int, default=FLOOR_TRAIN_IMAGES,
                   help="repeat smaller classes up to this many images (0 = off)")
    p.add_argument("--threshold-acc", type=float, default=THRESHOLD_ACC)
    p.add_argument("--init", metavar="CKPT",
                   help="start from this checkpoint's weights (fine-tuning)")
    p.add_argument("--out", "--checkpoint", dest="checkpoint", default=MODEL_PATH)
    args = p.parse_args()
    args.device_obj = get_device(args.device)

    model, classes, best, loaders = train(args)
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    # Saved before the diagnostics, so a failure there cannot lose the run.
    save_checkpoint(args.checkpoint, state, classes, args, val_metrics=best)

    report(VAL_SPLIT, best, classes)
    confusion_report(best, classes)
    val_probs, val_preds, val_targets = collect(model, loaders[VAL_SPLIT], args.device_obj)
    thresholds = threshold_sweep(val_probs, val_preds, val_targets, classes,
                                 args.threshold_acc)
    apply_thresholds(VAL_SPLIT, val_probs, val_preds, val_targets, classes, thresholds)
    prototype_report(model, loaders[VAL_SPLIT], args.device_obj, classes)
    save_checkpoint(args.checkpoint, state, classes, args, val_metrics=best,
                    thresholds=thresholds)

    # Weights and thresholds are frozen; test is read once and reported as-is.
    print(f"\n--- {TEST_SPLIT} (read once, nothing tuned on it) ---")
    probs, preds, targets = collect(model, loaders[TEST_SPLIT], args.device_obj)
    test = scores(preds, targets, len(classes))
    report(TEST_SPLIT, test, classes)
    confusion_report(test, classes)
    apply_thresholds(TEST_SPLIT, probs, preds, targets, classes, thresholds)
    print(f"\nsaved to {args.checkpoint}")


if __name__ == "__main__":
    main()
