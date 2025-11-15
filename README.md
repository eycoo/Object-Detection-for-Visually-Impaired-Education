# Object-Detection-for-Visually-Impaired-Education
Object Detection for Visually Impaired Education: tool to help visually impaired children learn programming logic. Using Object Detection, the system 'reads' physical flowchart blocks arranged by the student , allowing them to learn independently without needing visual cues.

# Flowchart Symbol Detection System 🚀

![Project Status](https://img.shields.io/badge/status-active-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Model](https://img.shields.io/badge/YOLOv8-ONNX%20INT8-orange)

A lightweight, real-time object detection module designed to recognize flowchart symbols. This model is optimized using **ONNX quantization (INT8)** to ensure high performance and low latency, making it suitable for deployment on edge devices or standard CPUs.

## 📂 Project Structure

This repository contains the core files for the detection module:

| File Name | Description |
| :--- | :--- |
| `farras_int8.onnx` | The trained YOLOv8 model exported to ONNX format with **INT8 Quantization** for faster inference speed. |
| `main2.py` | The main script for running **real-time detection** using a webcam/video feed. |
| `train.ipynb` | Jupyter Notebook containing the full training pipeline (dataset loading, training, and export). |

## 🧠 The Model

The model is based on the **YOLOv8** architecture and has been trained to detect the following flowchart components:

* **Start/End** (Terminator)
* **Process** (Rectangle)
* **Decision** (Diamond)
* **Input/Output** (Parallelogram)
* *(Add other classes here if any, e.g., Arrow/Line)*

### Why ONNX INT8?
We use `farras_int8.onnx` instead of the standard `.pt` file to reduce model size and increase inference speed (FPS) by up to 3x on CPU environments without significant loss in accuracy.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/username/project-name.git](https://github.com/username/project-name.git)
    cd project-name
    ```

2.  **Install dependencies:**
    You will need OpenCV, ONNX Runtime, and standard Python libraries.
    ```bash
    pip install opencv-python onnxruntime numpy
    ```
    *(Note: `ultralytics` is required only if you want to retrain using `train.ipynb`)*

## 💻 Usage

### 1. Running Real-time Detection
To test the model using your computer's webcam, simply run the main script:

```bash
python main2.py
