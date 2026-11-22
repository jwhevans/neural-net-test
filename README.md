# CNN Test - Tiny shape classifier

A ~28.6k-parameter CNN (PyTorch) that classifies 64×64 images as **circle, square,
triangle, star or rounded square** (squircles and rounded rectangles). It trains on
synthetic images in about a minute on a laptop
(it uses Apple-Silicon `mps` or CUDA when one is available, otherwise the CPU).

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python generate_data.py        # 2000 train + 400 val + 400 test images per class -> data/
python train.py                # trains, reports val + test metrics -> shape_cnn.pt
python predict.py path/to/image.png
```

`train.py` keeps the epoch with the best validation macro-F1, tunes a per-class
confidence threshold on val, and only then evaluates the test split once.
`predict.py` prints the class probabilities and whether the top prediction clears
its class's threshold (otherwise it reports `UNSURE`).

Useful flags:
- `generate_data.py --n-train/--n-val/--n-test/--seed/--out`,
  `--splits train` (redraw only some splits), `--force` (required to overwrite
  a split that already has images).
- `train.py --epochs/--lr/--batch-size/--weight-decay/--label-smoothing/--seed`,
  `--no-augmentation`, `--device cpu|mps|cuda`, `--out`,
  `--init shape_cnn.pt` (fine-tune from a checkpoint),
  `--balanced-sampling/--cap-train-images/--floor-train-images` (for imbalanced data).
- `predict.py --weights`.

## Files
- `config.py` – shared settings (classes, image size, normalisation, paths) and the defaults for every flag.
- `generate_data.py` – draws random shapes (color, size from 18–100% of the frame, position, rotation, filled/outline, noise). Each split has its own seed stream and is described in `data/manifest.json`.
- `model.py` – `TinyCNN`: 4 × (conv3×3 → BatchNorm → ReLU → maxpool), global avg pool, linear.
- `train.py` – trains with AdamW + cosine LR, label smoothing and flip/rot90/colour-jitter augmentation, reading images from disk with DataLoaders. Saves weights plus class names, preprocessing settings and thresholds in the checkpoint.
- `predict.py` – classifies one image and prints class probabilities. Refuses a checkpoint whose preprocessing doesn't match `config.py`.

## Adding a shape
Add a name to `CLASSES` in `config.py` and a drawing branch in `draw_shape` in
`generate_data.py`, then regenerate (`--force`) and retrain. The number of classes
is read from the data folders.
