"""Single-image prediction against the trained VGG16 classifier.

Paths are anchored to the repo root, not the working directory, so this runs from anywhere:

    python3 src/test.py                                  # default sample image
    python3 src/test.py data/test/dog/00500-3846168662.png
    python3 src/test.py img.png --model models/cat_vs_dog_vgg16_best.keras
"""

import argparse
import os
import pathlib

# Must be set before TensorFlow is imported: hides the AVX2/FMA and oneDNN notices.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input  # type: ignore

from data_loader import IMG_SIZE, OUT_DIR, ROOT, TEST_DIR
""""loads model vgg16 and then predict the image is cat or dog""""

DEFAULT_MODEL = OUT_DIR / "cat_vs_dog_vgg16.keras"
""""images path""""
DEFAULT_IMAGE = r'/mnt/d/Cat vs Dog/Cat-vs-Dog-classification/download.jpg'


def _resolve(path):
    """Accept absolute paths, or paths given relative to cwd or to the repo root."""
    path = pathlib.Path(path)
    if path.is_absolute():
        return path
    for base in (pathlib.Path.cwd(), ROOT):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (ROOT / path).resolve()


def predict(model, image_path):
    """Returns P(dog) for one image, preprocessed exactly like the training data."""
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    x = preprocess_input(np.expand_dims(tf.keras.utils.img_to_array(img), axis=0))
    return float(model.predict(x, verbose=0)[0][0])  # single sigmoid = P(dog)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", default=DEFAULT_IMAGE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    model_path, image_path = _resolve(args.model), _resolve(args.image)

    if not model_path.exists():
        available = sorted(p.name for p in OUT_DIR.glob("*.keras"))
        raise SystemExit(
            f"model not found: {model_path}\n"
            f"available in {OUT_DIR}: {available or 'none — run src/train.py first'}"
        )
    if not image_path.exists():
        raise SystemExit(f"image not found: {image_path}")

    model = tf.keras.models.load_model(model_path)
    p = predict(model, image_path)

    print(f"{image_path.name}: " + (f"Dog ({p:.2%})" if p >= 0.5 else f"Cat ({1 - p:.2%})"))


if __name__ == "__main__":
    main()