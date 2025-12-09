# ML Assignment: Explainable AI for Robot Telemetry Data

## Student Note
As per the instructor’s guideline, students with roll numbers ending in an even digit are required to implement **even-numbered models only**.  
Therefore, this submission includes the following models:

- 1D Convolutional Neural Network (1D-CNN)  
- Feedforward Neural Network (FNN)  
- XGBoost Classifier  

---

## Overview
This repository contains the complete implementation, trained models, results, and report for the Machine Learning assignment titled **"Explainable AI for Robot Telemetry Data"**.  
The objective of this assignment is to analyze and model telemetry data from drone or robotic systems, detect abnormal system behavior, and apply Explainable AI (XAI) methods to understand model predictions.  

The models classify system states into three categories:

- Normal  
- DoS Attack  
- Malfunction  

---

## Repository Contents

| File/Folder | Description |
| ----------- | ----------- |
| `ML_assignment_03.ipynb` | Main Jupyter Notebook containing complete code for preprocessing, training, evaluation, and explainability |
| `/models/` | Folder containing trained model files: `model_cnn.h5`, `model_fnn.h5`, `model_xgboost.pkl` |
| `/results/` | Folder containing result visualizations, confusion matrices, and SHAP explainability plots |
| `Report_Explainable_AI.pdf` | Final written report covering methodology, results, and analysis |
| `README.md` | Instructions for reproducing and running this assignment |

---

## Requirements
The following Python libraries are required (available by default in Google Colab):

- numpy  
- pandas  
- matplotlib  
- seaborn  
- scikit-learn  
- tensorflow  
- keras  
- xgboost  
- shap  
- lime  

---

## How to Run the Code

### Option 1: Open in Google Colab
1. Upload `ML_assignment_03.ipynb` to Google Colab.  

### Option 2: Open from GitHub
1. File → Open Notebook → GitHub → Paste repository link  

### Upload Required Files
Before running the notebook, upload the following files when prompted in Google Colab.

#### **Option 1: Run Full Training (Recommended)**
If you want to train the models from scratch, upload **only the dataset files** listed below when prompted:

**Dataset files:**  
- `Dos1.csv`, `Dos2.csv`  
- `Malfunction1.csv`, `Malfunction2.csv`  
- `Normal1.csv`, `Normal2.csv`, `Normal3.csv`, `Normal4.csv`  

The notebook will handle all preprocessing, training, evaluation, and explainability steps automatically.

#### **Option 2: Use Pre-Trained Models**
If you prefer to skip training and directly evaluate pre-trained models, upload the following **model files** in addition to the dataset:

**Model files:**  
- `model_cnn.h5`  
- `model_fnn.h5`  
- `model_xgboost.pkl`  

These models can be loaded directly to view performance metrics, confusion matrices, and explainability outputs without retraining.

---

### Run Notebook
1. Open the notebook in Google Colab.  
2. Upload the dataset (and model files if using pre-trained models) when prompted.  
3. Go to `Runtime` → `Run All` to execute all cells sequentially.  
4. The notebook will automatically display:  
   - Accuracy, Precision, Recall, F1-scores  
   - Confusion Matrices  
   - SHAP Summaries and Dependence Plots  
   - PDPs and Feature Importance Charts  

All major outputs will also be saved in the `/results/` folder and summarized in the report.

---

## Expected Results Summary

| Model | Accuracy | F1-Score | Remarks |
| ----- | -------- | -------- | ------- |
| 1D-CNN | 98.4% | 0.98 | Excellent temporal pattern recognition |
| FNN | 97.8% | 0.98 | Stable, consistent performance |
| XGBoost | 100% | 1.00 | Strong baseline model for structured data |

**Key influential features identified by explainability methods:**  
Battery Voltage, RSSI Signal, CPU Usage  

---

## Explainable AI Techniques
- **SHAP (SHapley Additive Explanations)**  
  - Global feature importance plots  
  - Beeswarm and dependence plots  
  - Feature-level impact interpretation  
- **Partial Dependence Plots (PDP)**  
  - Relationships between individual telemetry features and prediction outcomes  
- **Feature Importance (XGBoost)**  
  - Top 15 most impactful features based on gain scores  

---
## Submission Package Includes

| Component | Description |
| --------- | ----------- |
| `ML_assignment_03.ipynb` | Fully implemented and commented code notebook |
| `/models/` | Saved trained models (.h5 and .pkl formats) |
| `/results/` | Confusion matrices, accuracy charts, SHAP plots |
| `/dataset/` | All telemetry dataset files: `Dos1.csv`, `Dos2.csv`, `Malfunction1.csv`, `Malfunction2.csv`, `Normal1.csv`, `Normal2.csv`, `Normal3.csv`, `Normal4.csv` |
| `Report_Explainable_AI.pdf` | 10-page report with methodology, figures, and discussion |
| `README.md` | Instructions and reproduction steps |


---

## Reproducibility Notes
- Random seeds (`random_state=42`) used for reproducibility  
- Stratified train-test splits to maintain class balance  
- Preprocessing and scaling applied consistently before training  
- Saved models can be reloaded directly without retraining  

---

## Evaluation Checklist

| Requirement | Status |
| ----------- | ------ |
| Data Preprocessing & Cleaning | Completed |
| Model Implementation (Even Models Only) | Completed |
| Model Evaluation Metrics | Completed |
| Explainable AI Analysis (SHAP, PDP) | Completed |
| Report (10 Pages) | Completed |
| Trained Model Files | Included |
| Visualizations Folder | Included |
| README File | Included |

---

## Author
**Name:** Syed Muhammad Sheeraz Nadeem  
**Course:** Advance Artificial Intelligence  
**Assignment:** ML Assignment (Explainable AI for Robot Telemetry Data)  
**Instructor:** Sir Abdullah  
**Institution:** National University of Computer and Emerging Sciences (NUCES)  
