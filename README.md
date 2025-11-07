# Smart-Sort: AI-Powered Waste Classifier ♻️


## Live Application

**Try the live application here:** https://wasteclassifierapp.streamlit.app/
<!-- TODO: Replace with your actual deployed Streamlit app URL -->

---


## Overview

Smart-Sort is an end-to-end deep learning project that classifies waste into six distinct categories: **cardboard, glass, metal, paper, plastic, and trash**. The goal of this project is to create an intelligent system that can help automate and improve recycling processes.

A user can upload an image of a waste item, and the trained model will predict its category in real-time. The entire application is built with Python and deployed on the Streamlit Community Cloud.

---

## Features

- **Multi-Class Image Classification:** Utilizes a Convolutional Neural Network (CNN) to classify images.
- **Transfer Learning:** Built upon the powerful MobileNetV2 architecture, pre-trained on ImageNet, for high accuracy and efficiency.
- **Interactive Web Interface:** A user-friendly web app built with Streamlit allows for easy image uploads and displays clear predictions with confidence scores.
- **Modular and Reproducible Code:** The project is structured with separate scripts for data loading, model building, training, and evaluation, promoting clarity and reusability.
- **Cloud Deployed:** The application is live and accessible to anyone, anywhere.

---

## Tech Stack

- **Backend & Modeling:** Python, TensorFlow, Keras
- **Web Framework:** Streamlit
- **Data Handling:** NumPy, OpenCV, Pillow
- **Environment:** Conda (for local setup)
- **Deployment:** Streamlit Community Cloud, GitHub


---

## Installation and Usage

Follow these steps to set up and run the project on your local machine.

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- NVIDIA GPU with appropriate drivers and CUDA setup (for GPU acceleration)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda create --name smartsort python=3.10
    conda activate smartsort
    ```

3.  **Install the required packages:**
    The `requirements.txt` is for deployment. For a local setup with GPU support, install the following:
    ```bash
    # For NVIDIA GPU users (recommended)
    pip install tensorflow[and-cuda]

    # For CPU-only users
    pip install tensorflow

    # Install other necessary libraries
    pip install numpy opencv-python matplotlib scikit-learn jupyterlab streamlit Pillow
    ```

4.  **Download the Dataset:**
    - Download the **TrashNet** dataset (e.g., from [this GitHub repository](https://github.com/garythung/trashnet)).
    - Extract the contents into a `data/` folder in the root of the project directory. The `data/` folder should contain the 6 sub-folders of image categories.

### Training the Model

To train the model from scratch, run the `train.py` script. This will process the data, build the model, train it for 20 epochs, and save the final `waste_classifier_model.keras` file in the `models/` directory.

```bash
python train.py
```
---
## Model Performance
The model was evaluated on a held-out test set, achieving the following performance:<br>
  - Test Loss: 0.4818
  - Test Accuracy: 82.81% <br>
This demonstrates the model's strong ability to generalize to new, unseen images of waste. The training history shows a healthy learning curve with a good convergence between training and validation accuracy.

---
## Future Improvements
This project serves as a strong baseline, and several areas could be explored for future enhancement:
- **Improve Model Accuracy**: Experiment with other pre-trained architectures (e.g., EfficientNet, ResNet), fine-tuning of layers, and more advanced data augmentation techniques.
- **Expand the Dataset**: Collect more images, especially for the underrepresented trash category, to improve model robustness and reduce class imbalance.
- **Real-Time Object Detection**: Upgrade the model from a simple classifier to an object detection model (like YOLO or SSD) that can identify and classify multiple waste items in a single frame from a video stream.
- **IoT Integration**: Integrate the model with a physical smart bin using a Raspberry Pi and a camera to create a fully automated sorting system.
