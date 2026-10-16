
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image


project_dir = Path(__file__).resolve().parent.parent

dataset_path = (
    project_dir
    / "data"
    / "raw"
    / "Intel_Image_Classification"
)

train_path = dataset_path / "seg_train_test" / "seg_train"
pred_path = dataset_path / "seg_pred"

source_dir = train_path
destination_dir = (
    project_dir / "data" / "processed" / "images_150x150" / "seg_train"
)

prediction_source_dir = pred_path
prediction_destination_dir = (
    project_dir / "data" / "processed" / "images_150x150" / "seg_pred"
)

image_size = (150, 150)
extensions = (".jpg", ".jpeg", ".png")

class_names = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
]

sub_paths = [train_path / name for name in class_names]


def count_images(sub_paths, pred_path):
    class_counts = {}

    for class_dir in sub_paths:
        if not class_dir.is_dir():
            continue

        images = [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ]

        class_counts[class_dir.name] = len(images)

    prediction_images = [
        p for p in pred_path.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]

    print(f"Prediction images: {len(prediction_images)}")

    for name, count in class_counts.items():
        print(f"{name}: {count} images")

    print(f"Total training images: {sum(class_counts.values())}")


def visualise_images(sub_paths):
    fig, axes = plt.subplots(6, 4, figsize=(18, 16))

    for index, class_dir in enumerate(sub_paths):
        images = [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ]

        for i, image_path in enumerate(images[:4]):
            with Image.open(image_path) as img:
                axes[index, i].imshow(img.copy())

            axes[index, i].axis("off")
            axes[index, i].set_title(class_dir.name, fontsize=12)

    plt.tight_layout()
    plt.show()


def check_image_dimensions_and_modes(sub_paths):
    for class_dir in sub_paths:
        dimensions = set()
        modes = set()

        images = [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ]

        for image_path in images[:20]:
            with Image.open(image_path) as img:
                dimensions.add(img.size)
                modes.add(img.mode)

        print(f"{class_dir.name} dimensions: {dimensions}")
        print(f"{class_dir.name} modes: {modes}")


def check_image_formats(sub_paths):
    for class_dir in sub_paths:
        formats = {
            p.suffix.lower()
            for p in class_dir.iterdir()
            if p.is_file()
        }

        print(f"{class_dir.name}: {formats}")


def resize_training_images(source_dir, destination_dir, image_size):
    total_resized = 0

    for class_dir in source_dir.iterdir():
        if not class_dir.is_dir():
            continue

        output_class_dir = destination_dir / class_dir.name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        for image_path in class_dir.iterdir():
            if (
                not image_path.is_file()
                or image_path.suffix.lower() not in extensions
            ):
                continue

            with Image.open(image_path) as img:
                resized_img = img.resize(image_size)
                output_path = output_class_dir / image_path.name
                resized_img.save(output_path)

            total_resized += 1

    print(f"Training images resized: {total_resized}")


def resize_prediction_images(source_dir, destination_dir, image_size):
    destination_dir.mkdir(parents=True, exist_ok=True)
    total_resized = 0

    for image_path in source_dir.iterdir():
        if (
            not image_path.is_file()
            or image_path.suffix.lower() not in extensions
        ):
            continue

        with Image.open(image_path) as img:
            resized_img = img.resize(image_size)
            output_path = destination_dir / image_path.name
            resized_img.save(output_path)

        total_resized += 1

    print(f"Prediction images resized: {total_resized}")


def verify_resized_images(source_dir, destination_dir):
    all_valid = True

    for class_dir in source_dir.iterdir():
        if not class_dir.is_dir():
            continue

        original_images = [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ]

        resized_class_dir = destination_dir / class_dir.name

        if not resized_class_dir.is_dir():
            print(f"{class_dir.name}: destination folder missing")
            all_valid = False
            continue

        resized_images = [
            p for p in resized_class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ]

        dimensions = set()

        for image_path in resized_images:
            with Image.open(image_path) as img:
                dimensions.add(img.size)

        counts_match = len(original_images) == len(resized_images)
        dimensions_match = (
            not resized_images or dimensions == {image_size}
        )

        print(f"\n{class_dir.name}:")
        print(f"Original images: {len(original_images)}")
        print(f"Resized images: {len(resized_images)}")
        print(f"Dimensions: {dimensions}")
        print(f"Counts match: {counts_match}")
        print(f"All dimensions correct: {dimensions_match}")

        if not counts_match or not dimensions_match:
            all_valid = False

    print(f"\nTraining verification passed: {all_valid}")


def verify_prediction_images(source_dir, destination_dir):
    original_images = [
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]

    if not destination_dir.is_dir():
        print("Prediction destination folder does not exist.")
        return

    resized_images = [
        p for p in destination_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]

    dimensions = set()

    for image_path in resized_images:
        with Image.open(image_path) as img:
            dimensions.add(img.size)

    counts_match = len(original_images) == len(resized_images)
    dimensions_match = (
        not resized_images or dimensions == {image_size}
    )

    print("\nPrediction verification:")
    print(f"Original images: {len(original_images)}")
    print(f"Resized images: {len(resized_images)}")
    print(f"Dimensions: {dimensions}")
    print(f"Counts match: {counts_match}")
    print(f"All dimensions correct: {dimensions_match}")


if __name__ == "__main__":
    if not train_path.is_dir():
        raise FileNotFoundError(f"Training folder not found: {train_path}")

    if not pred_path.is_dir():
        raise FileNotFoundError(f"Prediction folder not found: {pred_path}")

    count_images(sub_paths, pred_path)
    check_image_dimensions_and_modes(sub_paths)
    check_image_formats(sub_paths)
    visualise_images(sub_paths)

    resize_training_images(
        source_dir,
        destination_dir,
        image_size,
    )

    resize_prediction_images(
        prediction_source_dir,
        prediction_destination_dir,
        image_size,
    )

    verify_resized_images(
        source_dir,
        destination_dir,
    )

    verify_prediction_images(
        prediction_source_dir,
        prediction_destination_dir,
    )
