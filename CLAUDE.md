# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small PyTorch training exercise: `TinyCNN` (~28.6k params) classifies 64×64 synthetic images into 5 shape classes (circle, square, triangle, star, rounded_square). Plain scripts, no package, no test suite or linter.

## Commands

```bash
source .venv/bin/activate            # deps: torch, torchvision, pillow, numpy (requirements.txt)
python generate_data.py              # writes data/{train,val,test}/<class>/NNNN.png + data/manifest.json
python generate_data.py --splits train --n-train 4000 --force   # redraw one split only
python train.py                      # trains, saves shape_cnn.pt
python predict.py path/to/image.png [--weights shape_cnn.pt]
```

`data/`, `sample/` (real-world test images), `*.pt` and `.venv/` are gitignored.

## Architecture

**`config.py` is the single source of truth** for anything the scripts must agree on (`CLASSES`, `SIZE`, `MEAN`/`STD`, split names, `DATA_DIR`, `MODEL_PATH`) and for every CLI default in `train.py`/`generate_data.py`. Change values there, not in the scripts.

**Data generation** (`generate_data.py`): each split draws from its own RNG seeded by `"<seed>:<split>"`, so changing one split's count leaves the others byte-identical. Redrawing a non-empty split is refused without `--force` (which deletes its old PNGs first, so ImageFolder never trains on stale images). Splits not named by `--splits` are untouched but still recorded in the manifest from on-disk counts. Shape span is bounded by `MIN_SPAN`/`MAX_SPAN` and shapes always stay inside the frame.

**Training** (`train.py`) loads splits via `torchvision.datasets.ImageFolder`. Split discipline is deliberate and should be preserved:
- `train` — the only split the weights see.
- `val` — model selection (best epoch by macro-F1), per-class acceptance-threshold tuning (target `THRESHOLD_ACC`), and diagnostics (prototype/cosine report).
- `test` — evaluated once, after weights are frozen; never used to tune anything.

Augmentation is restricted to lossless transforms (h/v flips, 90° rotations, bounded colour jitter). Don't add crops, translations or arbitrary-angle rotations — they would clip shapes the generator kept in-frame. Optional rebalancing (cap/floor, `WeightedRandomSampler`) is off by default because the generated set is balanced.

**Checkpoint format**: `save_checkpoint` stores a dict with `state_dict`, `classes`, `version`, `arch`, a `preprocess` block (`size`/`mean`/`std`), `train_config`, `val_metrics` and `thresholds` (per-class confidence bars, indexed like `classes`). `load_checkpoint`/`check_preprocess` live in `train.py` and are reused by `predict.py`, which refuses mismatched preprocessing, names outputs via `ckpt["classes"]` and flags predictions below their class threshold as UNSURE. `--init` fine-tunes from a checkpoint (classes must match).

**Class order gotcha**: the model's output index order is ImageFolder's (alphabetical folder names), *not* the order of `config.CLASSES`. Always map logits through the checkpoint's `classes`. The number of classes comes from the data folders.

**Inference** (`predict.py`): flattens transparency onto white, letterboxes non-square images using the median border colour as padding (so circles aren't stretched), then normalises with the shared `MEAN`/`STD`.

## Adding a shape

Add the name to `CLASSES` in `config.py`, add a drawing branch in `draw_shape` in `generate_data.py`, then regenerate (`--force`) and retrain.
