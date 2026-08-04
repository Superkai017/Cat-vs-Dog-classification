"""Train the VGG16 cat-vs-dog classifier and save it.

Mirrors the pipeline in `cat vs dog.ipynb` (same seed, architecture and epochs) so the
numbers are comparable, but runs headless and persists the artifacts:

    models/cat_vs_dog_vgg16.keras       final model (last epoch)
    models/cat_vs_dog_vgg16_best.keras  best epoch by val_accuracy
    models/history.json                 per-epoch training history + config
    models/training_curves.png

Training only touches data/train — the test set is never loaded here. Score the saved
models with `src/evaluate.py`:

    ~/tf-env/bin/python src/train.py
    ~/tf-env/bin/python src/evaluate.py
    ~/tf-env/bin/python src/evaluate.py --model models/cat_vs_dog_vgg16_best.keras \\
        --prefix best_val
"""
""" External Tool imported """
import json

import matplotlib

matplotlib.use("Agg")  # headless: write figures to disk instead of opening a window

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16

from data_loader import BATCH_SIZE, IMG_SIZE, OUT_DIR, SEED, load_train_val
""""Models Settings on th VGG16 
    EPOCHS = 30
    LEARNING_RATE = 1e-4
""""
EPOCHS = 30
LEARNING_RATE = 1e-4


def build_model():
    """"Before modelling I performs Feature extraction using VGG16 and then add some layers to the model for classification""""
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    base_model = VGG16(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base_model.trainable = False   #in this layers we freeze the layers of VGG16 model and we will only train the new layers after it flattern then we added for classification

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    acc, val_acc = history.history["accuracy"], history.history["val_accuracy"]
    loss, val_loss = history.history["loss"], history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs_range, acc, label="Training Accuracy", marker="o")
    axes[0].plot(epochs_range, val_acc, label="Validation Accuracy", marker="o")
    axes[0].set_title("Training vs Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_range, loss, label="Training Loss", marker="o")
    axes[1].plot(epochs_range, val_loss, label="Validation Loss", marker="o")
    axes[1].set_title("Training vs Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_curves.png", dpi=120)
    plt.close(fig)

    gap = acc[-1] - val_acc[-1]
    print(f"\nFinal training accuracy:   {acc[-1]:.4f}")
    print(f"Final validation accuracy: {val_acc[-1]:.4f}")
    print(f"Accuracy gap: {gap:.4f}")
    return gap


def main():
    OUT_DIR.mkdir(exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)

    print("GPUs:", tf.config.list_physical_devices("GPU") or "none (CPU)")
    train_ds, val_ds, class_names = load_train_val()
    """"automatically saves your neural network model or its weights 
    to a file at regular intervals or when 
    performance improves during training """""
    model = build_model()
    model.summary()

    best_path = OUT_DIR / "cat_vs_dog_vgg16_best.keras"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        best_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=0
    )
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[checkpoint], verbose=2
    )

    gap = plot_history(history)

    final_path = OUT_DIR / "cat_vs_dog_vgg16.keras"
    model.save(final_path)
    print(f"\nSaved final model -> {final_path}")
    print(f"Saved best-val model -> {best_path}")

    history_path = OUT_DIR / "history.json"
    history_path.write_text(
        json.dumps(
            {
                "config": {
                    "img_size": list(IMG_SIZE),
                    "batch_size": BATCH_SIZE,
                    "seed": SEED,
                    "epochs": EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "base_model": "VGG16 (frozen, imagenet)",
                    "preprocessing": "tensorflow.keras.applications.vgg16.preprocess_input",
                },
                "class_names": class_names,
                "train_val_gap": float(gap),
                "history": {k: [float(v) for v in vs] for k, vs in history.history.items()},
            },
            indent=2,
        )
    )
    print(f"Saved history -> {history_path}")
    print("\nNext: score the saved models with src/evaluate.py")


if __name__ == "__main__":
    main()
