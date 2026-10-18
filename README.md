# Intel Image Classification with Deep Learning

## Project Overview

This project focuses on image classification using Deep Learning and Convolutional Neural Networks (CNNs) with TensorFlow and Keras.

The objective is to build, train, evaluate, and optimize a CNN capable of classifying natural and urban scene images into six categories. The project also includes a prediction function that can classify new images using a trained model.

## Objectives

- Explore and prepare an image dataset for Deep Learning.
- Understand and implement a Convolutional Neural Network (CNN).
- Train a CNN using TensorFlow and Keras.
- Evaluate model performance using classification metrics.
- Experiment with different CNN configurations to improve performance.
- Save the final model and use it to predict new images.

## Dataset

The project uses the **Intel Image Classification** dataset, which contains images belonging to six scene categories.

| Class | Description |
|---|---|
| Buildings | Buildings and architectural scenes |
| Forest | Forests and wooded landscapes |
| Glacier | Glaciers and icy landscapes |
| Mountain | Mountain landscapes |
| Sea | Sea and coastal scenes |
| Street | Streets and urban environments |

The dataset includes labeled training and testing images, as well as a separate collection of unlabeled images for prediction.

The images are resized to **150 × 150 pixels** and processed before being passed to the model.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Git and GitHub

## Project Structure

```text
image-detection-DL/
│
├── data/
│   ├── raw/
│   │   └── Intel_Image_Classification/
│   │       ├── seg_train/
│   │       ├── seg_test/
│   │       └── seg_pred/
│   │
│   └── processed/
│       └── images_150x150/
│
├── notebooks/
│   ├── cnn_model.py
│   └── predict_image.py
│
├── models/
│   ├── intel_image_cnn.keras
│   └── optimization/
│       └── deeper_cnn.keras
│
├── .gitignore
├── requirements.txt
└── README.md
```

*Note: The structure above represents the main project components. Adjust it to match the actual files and folders in your repository.*

## Methodology

### 1. Data Preparation

- Load and inspect the image dataset.
- Identify the six scene categories.
- Resize images to 150 × 150 pixels.
- Convert images into numerical arrays.
- Normalize pixel values to the range [0, 1].
- Prepare training, validation, and testing data.

### 2. CNN Model Development

A Convolutional Neural Network is used to automatically learn visual features from images.

The main concepts involved include:

- **Convolutional layers:** Extract features such as edges, textures, shapes, and more complex patterns.
- **Activation functions:** Introduce non-linearity into the network.
- **Pooling layers:** Reduce spatial dimensions and computational cost.
- **Dropout:** Help reduce overfitting.
- **Dense layers:** Use extracted features to classify images.
- **Softmax output:** Produces class probabilities for multiclass classification.

### 3. Model Optimization

Several CNN configurations are compared to investigate the effect of architectural and training choices on performance.

Experiments may include:

- Increasing the number of convolutional layers or filters.
- Adjusting Dropout rates.
- Changing the learning rate.
- Modifying batch size.
- Applying data augmentation.
- Testing different numbers of epochs.

The final model should be selected based on validation performance, generalization, and the results of the final test evaluation.

### 4. Model Evaluation

The model is evaluated using metrics such as:

- Accuracy
- Loss
- Precision
- Recall
- F1-score
- Confusion matrix

The original CNN achieved the following baseline results:

| Metric | Result |
|---|---:|
| Training accuracy | 85.39% |
| Validation accuracy | 87.67% |
| Test accuracy | 85.76% |
| Test loss | 0.4016 |
| Test samples | 2,556 |

These values correspond to the original CNN baseline. The optimized model's results should be reported separately once its final evaluation is complete.

### 5. Image Prediction

The prediction module loads a saved CNN and classifies a new image.

The prediction workflow is:

1. Load the trained model.
2. Load a new image.
3. Resize the image to 150 × 150 pixels.
4. Apply the same preprocessing used during training.
5. Generate predictions.
6. Identify the predicted class.
7. Display the predicted class and its probability.
8. Visualize the input image with its prediction.

Example output:

```text
Predicted class: forest
Probability: 94.72%
```

*Example output only; actual predictions and probabilities depend on the input image and trained model.*

## Installation

### Prerequisites

- Python 3.10 or another Python version compatible with the installed TensorFlow release.
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/BOUCHAIB-EL-HAIDI/A2_S2_B1_dL_image_classification.git
cd A2_S2_B1_dL_image_classification
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If the requirements file has not yet been created, install the main libraries:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Run Image Prediction

From the project root, execute:

```bash
python notebooks/predict_image.py
```

When prompted, provide the path to an image you want to classify.

For example:

```text
Enter the image path: C:\path\to\your\image.jpg
```

The script displays the predicted class, probability, and input image.

Ensure that the saved model exists at `models/optimization/deeper_cnn.keras` and that its input dimensions, preprocessing, and class-index mapping match the prediction code.

## Results and Future Improvements

Potential improvements include:

- Comparing the original and deeper CNN configurations.
- Applying data augmentation to improve generalization.
- Tuning hyperparameters systematically.
- Analyzing incorrectly classified images.
- Experimenting with transfer learning using architectures such as MobileNetV2.
- Developing a simple web interface with Streamlit.

## Conclusion

This project demonstrates the main stages of an image classification pipeline, from dataset preparation and CNN development to model optimization, evaluation, and prediction.

It provides practical experience with TensorFlow, Keras, computer vision, and Deep Learning model evaluation.

## Author

**Bouchaib El Haidi**

GitHub: [BOUCHAIB-EL-HAIDI](https://github.com/BOUCHAIB-EL-HAIDI)

---

*Academic Deep Learning project — Intel Image Classification.*
