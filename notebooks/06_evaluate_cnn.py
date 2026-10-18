
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

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

    class_names = [
        "buildings",
        "forest",
        "glacier",
        "mountain",
        "sea",
        "street"
    ]

    model = tf.keras.models.load_model(model_path)

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

    y_true = []
    y_pred = []
    misclassified_images = []
    misclassified_true = []
    misclassified_pred = []
    misclassified_confidence = []

    for images, labels in test_ds:
        predictions = model(images, training=False).numpy()
        true_labels = labels.numpy()
        predicted_labels = np.argmax(predictions, axis=1)

        y_true.extend(true_labels.tolist())
        y_pred.extend(predicted_labels.tolist())

        wrong_indices = np.where(true_labels != predicted_labels)[0]

        for index in wrong_indices:
            if len(misclassified_images) >= 12:
                break

            misclassified_images.append(images[index].numpy())
            misclassified_true.append(int(true_labels[index]))
            misclassified_pred.append(int(predicted_labels[index]))
            misclassified_confidence.append(
                float(predictions[index, predicted_labels[index]])
            )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\nTest Results")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"Correct predictions: {np.sum(y_true == y_pred)}")
    print(f"Incorrect predictions: {np.sum(y_true != y_pred)}")

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

    print("\nConfusion Matrix")
    print(matrix)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names
    )

    display.plot(cmap="Blues", xticks_rotation=45)
    plt.title("CNN Confusion Matrix - Test Dataset")
    plt.tight_layout()
    plt.show()

    print("\nRecall by Class")
    for index, class_name in enumerate(class_names):
        total = np.sum(matrix[index])
        correct = matrix[index, index]
        recall = correct / total if total else 0
        print(f"{class_name}: {recall:.2%} ({correct}/{total})")

    if misclassified_images:
        number_of_images = len(misclassified_images)
        columns = 4
        rows = (number_of_images + columns - 1) // columns

        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(14, 4 * rows)
        )
        axes = np.array(axes).reshape(-1)

        for index, axis in enumerate(axes):
            if index >= number_of_images:
                axis.axis("off")
                continue

            axis.imshow(
                np.clip(misclassified_images[index], 0, 1)
            )

            axis.set_title(
                f"True: {class_names[misclassified_true[index]]}\n"
                f"Predicted: {class_names[misclassified_pred[index]]}\n"
                f"Confidence: {misclassified_confidence[index]:.1%}"
            )
            axis.axis("off")

        figure.suptitle(
            "Misclassified Test Images",
            fontsize=16
        )
        plt.tight_layout()
        plt.show()
    else:
        print("No misclassified images found.")


if __name__ == "__main__":
    main()
