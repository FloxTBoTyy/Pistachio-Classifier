
# Pistachio Image Classification Pipeline

## Overview

This project provides a complete pipeline for pistachio image processing, feature extraction, and machine learning classification. It uses OpenCV to preprocess and segment pistachio images, extracts 16 morphological features, and trains models like Logistic Regression or Random Forest to classify pistachio types.

The code is modular and organized into folders for preprocessing, feature extraction, modeling, and utilities, making it easy to maintain and extend.

---

## Features

- Image loading and segmentation to isolate pistachio objects  
- Extraction of area, perimeter, axes, eccentricity, solidity, and other shape descriptors  
- Automated dataset creation by processing images from folder structures  
- Data normalization, train-test split, and multi-set evaluation (Train, Test, Virgin, Extra Virgin)  
- Model training with Logistic Regression (default) and easy extensibility for other classifiers  
- Visualization of confusion matrices and reporting of accuracy, precision, recall, F1-score, and ROC-AUC

---

## Installation

1. Clone the repository:  
git clone https://github.com/yourusername/pistachio-classification.git  
cd pistachio-classification

2. Create and activate a Python virtual environment (recommended):  
python3 -m venv venv  
source venv/bin/activate  *(On Windows use: venv\Scripts\activate)*

3. Install dependencies:  
pip install -r requirements.txt

---

## Usage

1. Prepare your data folder with pistachio images structured like:

data/  
    Pistachio_Image_Dataset/  
        Kirmizi_Pistachio/  
        Siirt_Pistachio/

2. Run the pipeline:  
python main.py

This will:  
- Extract features from images and create a CSV dataset  
- Train the ML model and evaluate it on multiple data splits  
- Show confusion matrix plots and print performance metrics

---

## Project Structure

project/  
├── data/                       # Raw images and datasets  
├── notebooks/                  # Jupyter notebooks for exploration  
├── src/  
│    ├── preprocessing/          # Data loading and splitting  
│    ├── feature_extraction/     # Image processing and feature extraction  
│    ├── modeling/               # Model training and evaluation  
│    └── utils/                  # Helper functions (if any)  
├── main.py                    # Entry point script  
├── requirements.txt           # Python dependencies  

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or contributions, please open an issue or contact Daniel Guevara.
