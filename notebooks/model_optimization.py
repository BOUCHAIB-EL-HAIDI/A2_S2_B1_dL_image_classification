
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))

from preprocessing_augmentation import (
    load_datasets,
    normalize_datasets,
    optimize_datasets
)


def build_cnn_model(input_shape, num_classes, filters, dropout_rate, learning_rate):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    for number_of_filters in filters:
        model.add(
            tf.keras.layers.Conv2D(
                number_of_filters,
                kernel_size=3,
                activation="relu",
                padding="same"
            )
        )
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


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

    input_shape = train_ds.element_spec[0].shape[1:]
    num_classes = 6

    configurations = [
        {
            "name": "baseline",
            "filters": [32, 64, 128],
            "dropout": 0.25,
            "learning_rate": 0.001
        },
        {
            "name": "higher_dropout",
            "filters": [32, 64, 128],
            "dropout": 0.45,
            "learning_rate": 0.001
        },
        {
            "name": "deeper_cnn",
            "filters": [32, 64, 128, 256],
            "dropout": 0.45,
            "learning_rate": 0.001
        }
    ]

    output_dir = project_dir / "models" / "optimization"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for config in configurations:
        print(f"\n{'=' * 50}")
        print(f"Training configuration: {config['name']}")
        print(f"{'=' * 50}")

        model = build_cnn_model(
            input_shape=input_shape,
            num_classes=num_classes,
            filters=config["filters"],
            dropout_rate=config["dropout"],
            learning_rate=config["learning_rate"]
        )

        model_path = output_dir / f"{config['name']}.keras"

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

        start_time = time.perf_counter()

        history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=20,
            callbacks=callbacks
        )

        training_time = time.perf_counter() - start_time

        best_epoch = min(
            range(len(history.history["val_loss"])),
            key=lambda epoch: history.history["val_loss"][epoch]
        )

        results.append({
            "configuration": config["name"],
            "filters": str(config["filters"]),
            "dropout": config["dropout"],
            "learning_rate": config["learning_rate"],
            "best_epoch": best_epoch + 1,
            "train_accuracy": history.history["accuracy"][best_epoch],
            "val_accuracy": history.history["val_accuracy"][best_epoch],
            "val_loss": history.history["val_loss"][best_epoch],
            "training_time_seconds": round(training_time, 2),
            "model_path": str(model_path)
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="val_loss",
        ascending=True
    ).reset_index(drop=True)

    results_path = output_dir / "comparison.csv"
    results_df.to_csv(results_path, index=False)

    print("\nConfiguration comparison:")
    print(
        results_df[
            [
                "configuration",
                "best_epoch",
                "train_accuracy",
                "val_accuracy",
                "val_loss",
                "training_time_seconds"
            ]
        ].to_string(index=False)
    )

    best_result = results_df.iloc[0]
    best_model_path = Path(best_result["model_path"])

    print(f"\nSelected configuration: {best_result['configuration']}")
    print(f"Validation accuracy: {best_result['val_accuracy']:.4f}")
    print(f"Validation loss: {best_result['val_loss']:.4f}")

    best_model = tf.keras.models.load_model(best_model_path)

    test_loss, test_accuracy = best_model.evaluate(
        test_ds,
        verbose=1
    )

    print("\nFinal test results for the selected model:")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(
        results_df["configuration"],
        results_df["val_accuracy"]
    )
    axes[0].set_title("Validation Accuracy Comparison")
    axes[0].set_ylabel("Accuracy")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(
        results_df["configuration"],
        results_df["val_loss"]
    )
    axes[1].set_title("Validation Loss Comparison")
    axes[1].set_ylabel("Loss")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.show()

    print(f"\nComparison saved to: {results_path}")
    print(f"Selected model saved at: {best_model_path}")


if __name__ == "__main__":
    main()



# Comparaison des résultats et choix du modèle final
# Le modèle deeper_cnn obtient les meilleurs résultats sur les données de validation,
# avec une accuracy de 85.44 % et une val_loss de 0.4260.
# Cependant, son accuracy sur le jeu de test est de 84.27 %, contre 85.76 %
# pour le modèle initial. Le choix final doit donc tenir compte des performances
# sur la validation, de la généralisation et des résultats d'évaluation.

