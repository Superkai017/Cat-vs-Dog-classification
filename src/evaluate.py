"""Load a saved cat-vs-dog model and report its performance on the test set.

All evaluation lives here — `train.py` only trains and saves. Use this to score a model
without retraining, or to score a different image folder:

    ~/tf-env/bin/python src/evaluate.py
    ~/tf-env/bin/python src/evaluate.py --model models/cat_vs_dog_vgg16_best.keras \\
        --prefix best_val
    ~/tf-env/bin/python src/evaluate.py --data data/test --threshold 0.6

Writes, for `--prefix P`:

    models/metrics_P.json       scalar metrics + confusion matrix
    models/evaluation_P.png     metric bars + confusion matrix
    models/roc_curve_P.png

Preprocessing comes from `data_loader`, so the input matches what the model was trained
on (caffe BGR, mean-subtracted) — feeding raw 0-255 RGB silently produces garbage.
"""

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")  # headless: write figures to disk instead of opening a window

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from data_loader import OUT_DIR, ROOT, load_test


def compute_metrics(model, ds, class_names, threshold=0.5):
    """Scalar metrics plus the raw ROC points and per-image probabilities."""
    y_true = np.concatenate([y.numpy() for _, y in ds], axis=0).flatten()
    y_prob = model.predict(ds, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    loss, keras_acc = model.evaluate(ds, verbose=0)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    results = {
        "test_loss": float(loss),
        "keras_accuracy": float(keras_acc),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(auc(fpr, tpr)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "class_names": class_names,
        "threshold": float(threshold),
        "n_test": int(len(y_true)),
    }
    return results, y_true, y_prob, y_pred, (fpr, tpr)


def print_report(results, y_true, y_pred, class_names, source):
    print(f"\n--- {source}  ({results['n_test']} images, threshold={results['threshold']}) ---")
    print(f"Loss:      {results['test_loss']:.4f}")
    print(f"Accuracy:  {results['accuracy']:.4f}  (keras: {results['keras_accuracy']:.4f})")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"AUC:       {results['roc_auc']:.4f}")
    print(f"\nConfusion matrix (rows=true {class_names}, cols=pred):")
    print(np.array(results["confusion_matrix"]))
    print("\n" + classification_report(y_true, y_pred, target_names=class_names, digits=4))


def print_misclassified(y_true, y_pred, y_prob, file_paths, class_names):
    """The images the model got wrong, so they can be eyeballed."""
    wrong = np.where(y_pred != y_true.astype(int))[0]
    if not len(wrong):
        print("No misclassified images.")
        return

    print(f"Misclassified ({len(wrong)}):")
    for i in wrong:
        print(
            f"  {pathlib.Path(file_paths[i]).name}: true={class_names[int(y_true[i])]} "
            f"pred={class_names[y_pred[i]]} p({class_names[1]})={y_prob[i]:.3f}"
        )


def plot_metrics(results, class_names, prefix, out_dir=OUT_DIR):
    """`prefix` namespaces the figures so evaluating several models doesn't overwrite them."""
    cm = np.array(results["confusion_matrix"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    values = [results["accuracy"], results["precision"], results["recall"], results["f1"]]
    bars = axes[0].bar(names, values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"Model Evaluation Metrics — {prefix} model (test set)")
    axes[0].set_ylabel("Score")
    axes[0].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center")

    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_xticks(range(len(class_names)), class_names)
    axes[1].set_yticks(range(len(class_names)), class_names)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1].text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.colorbar(im, ax=axes[1])
    fig.tight_layout()
    fig.savefig(out_dir / f"evaluation_{prefix}.png", dpi=120)
    plt.close(fig)


def plot_roc(results, roc_points, prefix, out_dir=OUT_DIR):
    fpr, tpr = roc_points

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {results['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random guess")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — Cat vs Dog Classifier ({prefix} model)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"roc_curve_{prefix}.png", dpi=120)
    plt.close(fig)


def evaluate(model, ds, class_names, file_paths=None, prefix="final", threshold=0.5,
             source="test set", plots=True, out_dir=OUT_DIR):
    """Full report for one model: scalar metrics, confusion matrix, ROC, figures."""
    results, y_true, y_prob, y_pred, roc_points = compute_metrics(
        model, ds, class_names, threshold
    )

    print_report(results, y_true, y_pred, class_names, source)
    if file_paths is not None:
        print_misclassified(y_true, y_pred, y_prob, file_paths, class_names)

    if plots:
        out_dir.mkdir(exist_ok=True)
        plot_metrics(results, class_names, prefix, out_dir)
        plot_roc(results, roc_points, prefix, out_dir)
        print(f"\nSaved figures -> {out_dir / f'evaluation_{prefix}.png'}, "
              f"{out_dir / f'roc_curve_{prefix}.png'}")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(OUT_DIR / "cat_vs_dog_vgg16.keras"))
    parser.add_argument("--data", default=str(ROOT / "data" / "test"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--prefix", default="final", help="namespaces the figures and metrics file"
    )
    parser.add_argument("--no-plots", action="store_true", help="skip writing the figures")
    args = parser.parse_args()

    ds, class_names, file_paths = load_test(args.data)

    model = tf.keras.models.load_model(args.model)
    print(f"Loaded {args.model}  ({model.count_params():,} params)")

    results = evaluate(
        model,
        ds,
        class_names,
        file_paths=file_paths,
        prefix=args.prefix,
        threshold=args.threshold,
        source=args.data,
        plots=not args.no_plots,
    )
    results["model"] = args.model
    results["data"] = args.data

    OUT_DIR.mkdir(exist_ok=True)
    metrics_path = OUT_DIR / f"metrics_{args.prefix}.json"
    metrics_path.write_text(json.dumps(results, indent=2))
    print(f"Saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
