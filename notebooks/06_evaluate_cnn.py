
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, str(Path(__file__).parent))

from preprocessing_augmentation import (
    load_datasets,
    normalize_datasets,
    optimize_datasets
)


def main():
    project_dir = Path(__file__).resolve().parent.parent
    model_path = project_dir / "models" / "intel_image_cnn.keras"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. Run the training script first."
        )

    train_ds, validation_ds, test_ds = load_datasets(project_dir)

    train_ds, validation_ds, test_ds = normalize_datasets(
        train_ds,
        validation_ds,
        test_ds
    )

    train_ds, validation_ds, test_ds = optimize_datasets(
        train_ds,
        validation_ds,
        test_ds
    )

    model = tf.keras.models.load_model(model_path)

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

    class_names = [
        "buildings",
        "forest",
        "glacier",
        "mountain",
        "sea",
        "street"
    ]

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model(images, training=False)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions.numpy(), axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\nTest Results")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    print("\nClassification Report")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=range(len(class_names)),
            target_names=class_names,
            zero_division=0
        )
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=range(len(class_names))
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names
    )

    display.plot(cmap="Blues", xticks_rotation=45)
    plt.title("CNN Confusion Matrix - Test Dataset")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
