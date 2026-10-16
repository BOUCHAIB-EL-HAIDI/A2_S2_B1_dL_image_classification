
import tensorflow as tf
from tensorflow.keras import layers, models
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from preprocessing_augmentation import create_data_augmentation


def create_cnn_model(input_shape=(150, 150, 3), num_classes=6):
    data_augmentation = create_data_augmentation()

    model = models.Sequential([
        layers.Input(shape=input_shape),

        data_augmentation,

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation="softmax")
    ], name="intel_image_cnn")

    return model


def compile_cnn_model():
    model = create_cnn_model()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    model = compile_cnn_model()
    model.summary()

    dummy_images = tf.zeros((1, 150, 150, 3))
    predictions = model(dummy_images, training=False)

    print("Input shape:", dummy_images.shape)
    print("Output shape:", predictions.shape)
    print("Number of classes:", predictions.shape[-1])


if __name__ == "__main__":
    main()
