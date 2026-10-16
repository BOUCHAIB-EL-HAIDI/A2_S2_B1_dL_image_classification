
from pathlib import Path
import shutil

from sklearn.model_selection import train_test_split

project_dir = Path(__file__).resolve().parent.parent
print(project_dir)
source_dir = project_dir / "data" / "processed" / "images_150x150" / "seg_train"
destination_dir = project_dir / "data" / "processed" / "split"

classes = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
]

image_paths = []
labels = []

for class_name in classes:
    class_dir = source_dir / class_name

    for image_path in class_dir.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            image_paths.append(image_path)
            labels.append(class_name)

train_paths, temporary_paths, train_labels, temporary_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.30,
    random_state=42,
    stratify=labels,
)

validation_paths, test_paths, validation_labels, test_labels = train_test_split(
    temporary_paths,
    temporary_labels,
    test_size=0.50,
    random_state=42,
    stratify=temporary_labels,
)

splits = {
    "train": (train_paths, train_labels),
    "validation": (validation_paths, validation_labels),
    "test": (test_paths, test_labels),
}

for split_name, (paths, split_labels) in splits.items():
    for image_path, class_name in zip(paths, split_labels):
        class_destination = destination_dir / split_name / class_name
        class_destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, class_destination / image_path.name)

    print(f"{split_name}: {len(paths)} images")

print(f"Total: {len(image_paths)} images")

