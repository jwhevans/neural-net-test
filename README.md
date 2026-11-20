# CNN Test - Tiny shape classifier

A ~28k-parameter CNN (PyTorch) that classifies 64×64 images as **circle, square,
triangle, star or rounded square** (squircles and rounded rectangles). It trains on
synthetic images in under a minute on a laptop
(it uses Apple-Silicon `mps` or CUDA when one is available, otherwise the CPU).

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python generate_data.py        # 2000 train + 400 val images per class -> data/
python train.py                # trains, prints val accuracy + confusion matrix -> shape_cnn.pt
python predict.py path/to/image.png
```

Useful flags: `generate_data.py --n-train/--n-val/--seed/--out`,
`train.py --epochs/--lr/--batch-size`.

## Files
- `generate_data.py` – draws random shapes (color, size from 10–100% of the frame, position, rotation, filled/outline, noise).
- `model.py` – `TinyCNN`: 4 × (conv3×3 → BatchNorm → ReLU → maxpool), global avg pool, linear.
- `train.py` – loads the dataset into memory, trains with Adam + cosine LR and flip/rot90 augmentation.
- `predict.py` – classifies one image and prints class probabilities.

## Adding a shape
Add a name to `CLASSES` in `generate_data.py` and a drawing branch in `draw_shape`,
then regenerate and retrain. The number of classes is read from the data folders.
