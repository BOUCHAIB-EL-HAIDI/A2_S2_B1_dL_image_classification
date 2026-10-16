
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))

from preprocessing_augmentation import (
    load_datasets,
    normalize_datasets,
    optimize_datasets
)
from cnn_model import compile_cnn_model


def main():
    project_dir = Path(__file__).resolve().parent.parent

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

    model = compile_cnn_model()

    output_dir = project_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "intel_image_cnn.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=20,
        callbacks=callbacks
    )

    model.save(model_path)

    print("\nTraining completed.")
    print("Model saved to:", model_path)
    print("Epochs completed:", len(history.history["loss"]))
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history["accuracy"], label="Training accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation accuracy")
    axes[0].set_title("CNN Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["loss"], label="Training loss")
    axes[1].plot(history.history["val_loss"], label="Validation loss")
    axes[1].set_title("CNN Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
