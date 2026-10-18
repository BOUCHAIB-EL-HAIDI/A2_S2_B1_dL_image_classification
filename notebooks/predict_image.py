
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "optimization" / "deeper_cnn.keras"

IMAGE_SIZE = (150, 150)

CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]


def predict_image(image_path):
    model = load_model(MODEL_PATH)

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )

    image_array = img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    probability = float(predictions[predicted_index]) * 100

    print(f"\nPredicted class: {predicted_class}")
    print(f"Probability: {probability:.2f}%")

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f"{predicted_class} - {probability:.2f}%")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return predicted_class, probability


if __name__ == "__main__":
    image_path = input("Enter the image path: ").strip().strip('"')
    predict_image(image_path)
