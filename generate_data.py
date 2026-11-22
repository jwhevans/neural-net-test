"""Generate synthetic labelled shape images for training.

Writes 64x64 PNGs to <out>/{train,val,test}/<class>/NNNN.png so that
torchvision.datasets.ImageFolder can load them directly.

Each split draws from its own RNG stream derived from --seed ("<seed>:<split>"),
so regenerating with a different --n-train leaves the val and test images
byte-identical, and no image can end up in two splits.

A run only redraws the splits named by --splits. Splits it does not name are not
opened at all, and their counts are carried into the manifest, so regenerating
one split never reports a dataset that lost the others.

Redrawing a split that already holds images is refused unless --force is given,
which deletes that split's images first. Without it, a smaller --n-train would
leave the surplus of the old, bigger run in place and ImageFolder would quietly
keep training on images this run never drew.
"""
import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

from config import CLASSES, DATA_DIR, N_TEST, N_TRAIN, N_VAL, SIZE

# Shape span (circumscribed-circle diameter) as a fraction of image width.
# 0.10 draws specks 6px across, which even a human cannot name reliably, so the
# lower bound sits where the shape is still legible.
MIN_SPAN, MAX_SPAN = 0.18, 1.00


def split_rng(seed, split):
    """A RNG stream that only depends on (seed, split), not on the other splits."""
    return random.Random(f"{seed}:{split}")


def random_color(rng):
    return tuple(rng.randint(0, 255) for _ in range(3))


def contrasting_colors(rng):
    """Pick a background and foreground colour that differ enough to see."""
    while True:
        bg, fg = random_color(rng), random_color(rng)
        if sum(abs(a - b) for a, b in zip(bg, fg)) > 200:
            return bg, fg


def polygon(cx, cy, radius, n_points, rotation, inner_ratio=None):
    """Vertices of a regular polygon (or a star if inner_ratio is given)."""
    pts = []
    steps = n_points * 2 if inner_ratio else n_points
    for i in range(steps):
        r = radius * inner_ratio if (inner_ratio and i % 2) else radius
        a = rotation + 2 * math.pi * i / steps
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def transform(local_pts, cx, cy, rotation):
    """Rotate shape-local (x, y) points and move them to (cx, cy)."""
    c, s = math.cos(rotation), math.sin(rotation)
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in local_pts]


def squircle(radius, rng, n_points=64):
    """Superellipse |x|^n + |y|^n = a^n, scaled to fit inside `radius`.

    n between 3.5 and 6 keeps it clearly between a circle (n=2) and a square.
    """
    n = rng.uniform(3.5, 6.0)
    # The farthest point is on the diagonal, at distance a * sqrt(2) * 2^(-1/n).
    a = radius / (math.sqrt(2) * 2 ** (-1 / n))
    pts = []
    for i in range(n_points):
        t = 2 * math.pi * i / n_points
        ct, st = math.cos(t), math.sin(t)
        pts.append((a * math.copysign(abs(ct) ** (2 / n), ct),
                    a * math.copysign(abs(st) ** (2 / n), st)))
    return pts


def rounded_rect(radius, rng, arc_points=12):
    """Square with circular corners, scaled to fit inside `radius`."""
    corner_frac = rng.uniform(0.2, 0.45)  # corner radius / half-side
    # Farthest point: (h - c) * sqrt(2) + c = radius, with c = corner_frac * h.
    h = radius / ((1 - corner_frac) * math.sqrt(2) + corner_frac)
    c = corner_frac * h
    pts = []
    for qx, qy, start in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
        ox, oy = qx * (h - c), qy * (h - c)  # corner arc centre
        for i in range(arc_points + 1):
            a = math.radians(start + 90 * i / arc_points)
            pts.append((ox + c * math.cos(a), oy + c * math.sin(a)))
    return pts


def draw_shape(shape, rng):
    bg, fg = contrasting_colors(rng)
    img = Image.new("RGB", (SIZE, SIZE), bg)
    d = ImageDraw.Draw(img)

    radius = rng.uniform(MIN_SPAN, MAX_SPAN) * SIZE / 2
    # Outlines on tiny shapes are just smudges, so keep those filled.
    filled = radius < 6 or rng.random() < 0.7
    width = rng.randint(1, max(1, min(4, int(radius / 5))))
    # A polygon outline is stroked centred on the edge, so half its width lies
    # outside `radius`; shrink the shape so the stroke stays inside the frame.
    if not filled:
        radius = min(radius, SIZE / 2 - width / 2)
    margin = radius if filled else radius + width / 2
    cx = rng.uniform(margin, SIZE - margin)
    cy = rng.uniform(margin, SIZE - margin)
    rot = rng.uniform(0, 2 * math.pi)
    kw = {"fill": fg} if filled else {"outline": fg, "width": width}

    if shape == "circle":
        d.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], **kw)
    else:
        if shape == "square":
            pts = polygon(cx, cy, radius, 4, rot)
        elif shape == "triangle":
            pts = polygon(cx, cy, radius, 3, rot)
        elif shape == "star":
            # Varying the inner ratio teaches the model "five points with a
            # concave outline" instead of one memorised star silhouette.
            pts = polygon(cx, cy, radius, 5, rot, inner_ratio=rng.uniform(0.35, 0.55))
        else:  # rounded_square: half squircles, half rounded rectangles
            local = squircle(radius, rng) if rng.random() < 0.5 else rounded_rect(radius, rng)
            pts = transform(local, cx, cy, rot)
        if filled:
            d.polygon(pts, fill=fg)
        else:
            d.line(pts + [pts[0]], fill=fg, width=width, joint="curve")

    # Light pixel noise so the net can't rely on perfectly flat colours.
    px = img.load()
    for _ in range(rng.randint(0, 150)):
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        px[x, y] = random_color(rng)
    return img


SPLITS = ("train", "val", "test")


def split_from_disk(out, split):
    """Count the images a split already has, so a partial run can still record it.

    Returns None when the split has nothing on disk. `per_class` is the largest
    class folder, which is what a run that drew every class at once leaves behind.
    """
    root = Path(out) / split
    if not root.is_dir():
        return None
    counts = {folder.name: len(list(folder.glob("*.png")))
              for folder in root.iterdir() if folder.is_dir()}
    counts = {name: n for name, n in counts.items() if n}
    if not counts:
        return None
    return {"per_class": max(counts.values()), "total": sum(counts.values())}


def prepare_folder(folder, force):
    """Return an empty folder to draw into.

    Refuses to overwrite an existing non-empty split unless --force is passed;
    with --force every old PNG is removed first, so a smaller image count can
    never leave stale images that ImageFolder would keep training on.
    """
    existing = sorted(folder.glob("*.png")) if folder.is_dir() else []
    if existing and not force:
        raise SystemExit(
            f"{folder} already holds {len(existing)} images; pass --force to "
            f"delete them and regenerate"
        )
    for path in existing:
        path.unlink()
    folder.mkdir(parents=True, exist_ok=True)


def generate(out, seed, counts, force=False):
    """Draw `counts[split]` images per class for the splits named in `counts`.

    Splits this run does not name are never opened, but they are still recorded:
    their counts come from what is on disk, which is what ImageFolder will read,
    not from whatever an earlier run happened to write into the manifest.
    """
    out = Path(out)
    manifest_path = out / "manifest.json"
    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            print(f"{manifest_path} was not readable; rebuilding it from disk")
            manifest = {}
    manifest["seed"] = seed
    manifest["size"] = SIZE
    manifest["classes"] = list(CLASSES)
    manifest["min_span"] = MIN_SPAN
    manifest["max_span"] = MAX_SPAN
    manifest["splits"] = {}
    for split in SPLITS:
        if split in counts:
            continue  # written below, from the images this run is about to draw
        on_disk = split_from_disk(out, split)
        if on_disk:
            manifest["splits"][split] = on_disk
            print(f"{split}: {on_disk['total']} images left untouched")
        else:
            print(f"{split}: nothing on disk and not drawn this run")
    for split, n in counts.items():
        rng = split_rng(seed, split)
        for shape in CLASSES:
            folder = out / split / shape
            prepare_folder(folder, force)
            for i in range(n):
                draw_shape(shape, rng).save(folder / f"{i:04d}.png")
        manifest["splits"][split] = {"per_class": n, "total": n * len(CLASSES)}
        print(f"{split}: {n} images x {len(CLASSES)} classes = {n * len(CLASSES)}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-train", type=int, default=N_TRAIN, help="images per class")
    p.add_argument("--n-val", type=int, default=N_VAL, help="images per class")
    p.add_argument("--n-test", type=int, default=N_TEST,
                   help="held-out test images per class (never used for training)")
    p.add_argument("--out", default=DATA_DIR,
                   help="dataset folder to write into (default: the one config.py "
                        "and train.py use, so the two never drift apart)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true",
                   help="delete the existing images of the splits being redrawn "
                        "first; other splits are never touched")
    p.add_argument("--splits", default="train,val,test", metavar="SPLIT[,...]",
                   help="which splits to (re)draw; the others are left untouched "
                        "on disk and still counted in the manifest")
    args = p.parse_args(argv)

    counts = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    wanted = [s.strip() for s in args.splits.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in counts]
    if unknown:
        p.error(f"unknown split(s): {', '.join(unknown)} "
                f"(choose from {', '.join(counts)})")
    generate(args.out, args.seed, {s: counts[s] for s in wanted}, force=args.force)
    print(f"manifest written to {Path(args.out) / 'manifest.json'}")


if __name__ == "__main__":
    main()
