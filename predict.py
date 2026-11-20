"""Classify a single image with the trained TinyCNN."""
import argparse

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from model import TinyCNN
from train import MEAN, STD

SIZE = 64


def border_color(img):
    """Median colour of the outermost pixels, i.e. the likely background."""
    w, h = img.size
    px = img.load()
    edge = [px[x, y] for x in range(w) for y in (0, h - 1)]
    edge += [px[x, y] for y in range(h) for x in (0, w - 1)]
    return tuple(sorted(c[i] for c in edge)[len(edge) // 2] for i in range(3))


def load_image(path):
    """Open any image and turn it into a 64x64 RGB image like the training data.

    - Transparent areas are flattened onto white (instead of turning black).
    - Non-square images are scaled to fit and padded with their own background
      colour, so circles stay round instead of being stretched into ovals.
    """
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    img = img.convert("RGB")
    return ImageOps.pad(img, (SIZE, SIZE), Image.LANCZOS, color=border_color(img))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image")
    p.add_argument("--weights", default="shape_cnn.pt")
    args = p.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu")
    classes = ckpt["classes"]
    model = TinyCNN(len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    x = tf(load_image(args.image)).unsqueeze(0)
    with torch.no_grad():
        probs = model(x).softmax(1)[0]

    best = probs.argmax().item()
    print(f"prediction: {classes[best]} ({probs[best]:.1%})")
    for c, pr in sorted(zip(classes, probs.tolist()), key=lambda t: -t[1]):
        print(f"  {c:10s} {pr:.3f}")


if __name__ == "__main__":
    main()
