"""Dataset configuration and loaders shared by `train.py` and `evaluate.py`.

Every script goes through here so the image size, batch size, seed and preprocessing
stay identical between training and evaluation — a model trained on VGG16-preprocessed
input silently produces garbage if it is later fed raw 0-255 RGB.

    from data_loader import load_train_val, load_test, IMG_SIZE
"""

import pathlib

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42
VALIDATION_SPLIT = 0.1

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUT_DIR = ROOT / "models"


def _image_dataset(directory, **kwargs):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        **kwargs,
    )


def _prepare(ds, shuffle=False):
    """VGG16 expects caffe-style BGR inputs, not [0,1] — preprocess before caching."""
    ds = ds.map(lambda x, y: (preprocess_input(x), y))
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(1000, seed=SEED)
    return ds.prefetch(tf.data.AUTOTUNE)


def load_train_val(directory=TRAIN_DIR):
    """Train/val are carved out of data/train with a shared seed, so the split is disjoint."""
    train_ds = _image_dataset(
        directory, validation_split=VALIDATION_SPLIT, subset="training", seed=SEED
    )
    val_ds = _image_dataset(
        directory, validation_split=VALIDATION_SPLIT, subset="validation", seed=SEED
    )

    overlap = set(train_ds.file_paths) & set(val_ds.file_paths)
    assert not overlap, f"train/val leakage: {len(overlap)} shared images"
    print(f"train: {len(train_ds.file_paths)}  val: {len(val_ds.file_paths)}")

    class_names = train_ds.class_names
    print("classes:", class_names)

    return _prepare(train_ds, shuffle=True), _prepare(val_ds), class_names


def load_test(directory=TEST_DIR):
    """Returns (dataset, class_names, file_paths); unshuffled so paths line up with predictions."""
    test_ds = _image_dataset(directory, shuffle=False)
    class_names, file_paths = test_ds.class_names, test_ds.file_paths
    print(f"test: {len(file_paths)}  classes: {class_names}")

    return _prepare(test_ds), class_names, file_paths
