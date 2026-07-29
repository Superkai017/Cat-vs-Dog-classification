# Cat-vs-Dog-classification
# 🐱🐶 Cat vs Dog Classification

My first image classification project — a binary image classifier built with **TensorFlow/Keras** that distinguishes between images of cats and dogs using a Convolutional Neural Network (CNN).

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project marks my first hands-on experience with deep learning and computer vision. It trains a CNN model to classify images as either a **cat** or a **dog**, covering the full pipeline: data preprocessing, model building, training, evaluation, and inference on new images.

## 🧠 Model Architecture

The model is a Convolutional Neural Network built with `tf.keras.Sequential`, typically consisting of:

- Multiple **Conv2D** + **MaxPooling2D** blocks for feature extraction
- **Dropout** layers to reduce overfitting
- A **Flatten** layer followed by **Dense** layers
- A final **Dense(1, activation='sigmoid')** output layer for binary classification

> Update this section with your exact architecture, layer sizes, and activation functions once finalized.

## 📂 Dataset

- **Source:** [Dog vs Cat — Anthony Therrien (Kaggle)](https://www.kaggle.com/datasets/anthonytherrien/dog-vs-cat)
- Downloaded programmatically via [`kagglehub`](https://github.com/Kaggle/kagglehub):

  ```python
  import kagglehub

  # Download latest version
  path = kagglehub.dataset_download("anthonytherrien/dog-vs-cat")
  print("Path to dataset files:", path)
  ```

- Images are split into `train/` and `test/` (or `validation/`) directories
- Preprocessing includes resizing, normalization (rescaling pixel values to `[0,1]`), and data augmentation (rotation, flip, zoom) using `ImageDataGenerator` or `tf.keras.utils.image_dataset_from_directory`

## 📁 Project Structure

```
Cat-vs-Dog-classification/
│
├── dataset/                  # Training and test images (not included in repo)
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   └── test/
│       ├── cats/
│       └── dogs/
│
├── model/                    # Saved trained model(s)
│   └── cat_dog_model.h5
│
├── notebooks/                 # Jupyter notebooks for experimentation
│   └── cat_dog_classification.ipynb
│
├── src/                       # Source scripts
│   ├── train.py
│   ├── predict.py
│   └── preprocess.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

> Adjust this tree to match your actual repo layout.

## ⚙️ Installation

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/Cat-vs-Dog-classification.git
   cd Cat-vs-Dog-classification
   ```

2. Create a virtual environment (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

### requirements.txt
```
tensorflow>=2.10
numpy
matplotlib
pillow
scikit-learn
kagglehub
```

## 🚀 Usage

### Train the model
```bash
python src/train.py
```

### Evaluate the model
```bash
python src/evaluate.py
```

### Predict on a new image
```bash
python src/predict.py --image path/to/image.jpg
```

Example (Python):
```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('model/cat_dog_model.h5')

img = image.load_img('path/to/image.jpg', target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
print("Dog 🐶" if prediction[0] > 0.5 else "Cat 🐱")
```

## 📊 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | XX% |
| Validation Accuracy | XX% |
| Test Accuracy | XX% |
| Loss | X.XX |

> Fill in with your actual results, and consider adding accuracy/loss curve plots here.

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook

## 🎯 Future Improvements

- [ ] Use transfer learning (e.g., MobileNetV2, VGG16, ResNet50) to boost accuracy
- [ ] Deploy the model as a web app (Flask/Streamlit)
- [ ] Add real-time prediction via webcam
- [ ] Expand to multi-class animal classification

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](../../issues).

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

## 🙋 Author

Built with curiosity as my first deep learning / computer vision project.

---

⭐ If you found this project helpful, consider giving it a star!
