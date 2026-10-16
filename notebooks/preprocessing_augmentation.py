
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt

#data loading phase 

def load_datasets(project_dir, image_size=(150, 150), batch_size=32, seed=42):
    split_dir = project_dir / "data" / "processed" / "split"

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        split_dir / "train",
        image_size=image_size,
        batch_size=batch_size,
        seed=seed
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        split_dir / "validation",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        split_dir / "test",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataset, validation_dataset, test_dataset

#normalisation phase 

def normalize_datasets(train_dataset, validation_dataset, test_dataset):
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

    train_dataset = train_dataset.map(
        lambda images, labels: (normalization_layer(images), labels)
    )

    validation_dataset = validation_dataset.map(
        lambda images, labels: (normalization_layer(images), labels)
    )

    test_dataset = test_dataset.map(
        lambda images, labels: (normalization_layer(images), labels)
    )

    return train_dataset, validation_dataset, test_dataset

#data augmentation

def create_data_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomContrast(0.1)
    ])


#augmented data vs orginal data

def visualize_augmentation(dataset, augmentation, class_names, num_images=5):
    images, labels = next(iter(dataset))

    plt.figure(figsize=(15, 6))

    for i in range(num_images):
        augmented_image = augmentation(images[i], training=True)

        plt.subplot(2, num_images, i + 1)
        plt.imshow(images[i].numpy())
        plt.title(f"Original: {class_names[labels[i].numpy()]}")
        plt.axis("off")

        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(tf.clip_by_value(augmented_image, 0, 1).numpy())
        plt.title("Augmented")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

#that optimisation preparing data (batches) for the model

def optimize_datasets(train_dataset, validation_dataset, test_dataset):
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset

def main():
    project_dir = Path.cwd()

    train_dataset, validation_dataset, test_dataset = load_datasets(project_dir)

    class_names = train_dataset.class_names

    train_dataset, validation_dataset, test_dataset = normalize_datasets(
        train_dataset,
        validation_dataset,
        test_dataset
    )

    data_augmentation = create_data_augmentation()

    visualize_augmentation(
        train_dataset,
        data_augmentation,
        class_names
    )
    optimize_datasets(train_dataset, validation_dataset, test_dataset)


if __name__ == "__main__":
    main()
