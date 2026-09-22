# Student Stress Detection System

An end-to-end Machine Learning web application designed to predict a student's stress level based on academic, lifestyle, physiological, and behavioral attributes.

---

## 📌 Project Overview

The **Student Stress Detection System** evaluates key student indicators—such as study hours, sleep duration, heart rate, and academic workload—to categorize stress into four distinct levels:
* **No Stress**
* **Low Stress**
* **Medium Stress**
* **High Stress**

The system encompasses data inspection, feature engineering, model training, pipeline serialization, and real-time deployment via an interactive Streamlit user interface.

---

## 📊 Dataset & Features

The model was trained on a synthetic dataset of 500 student records using 9 key input features:

1. **Sleep Hours**: Average daily sleep (hours)
2. **Study Hours**: Daily study time (hours)
3. **Heart Rate**: Average resting heart rate (bpm)
4. **Academic Workload**: Rating of academic load
5. **Caffeine Intake**: Daily caffeine consumption
6. **Physical Activity**: Weekly exercise/activity level
7. **Social Interaction**: Time spent socializing
8. **GPA**: Current Grade Point Average
9. **Perceived Control**: Subjective level of control over tasks

> *Note: `student_id` was removed during preprocessing as it serves only as a record identifier.*

---

## ⚙️ Technical Architecture & Pipeline

1. **Data Inspection & Preprocessing**:
   * Evaluated structure (`info()`), statistical summaries (`describe()`), and verified missing values.
   * Split data into **80% training** and **20% testing** sets.
2. **Feature Scaling**:
   * Applied `StandardScaler` to normalize numerical ranges across all features.
3. **Model Training**:
   * Trained a **Random Forest Classifier** with **300 decision trees** to ensure high generalization accuracy and resistance to overfitting.
4. **Model Evaluation**:
   * Measured performance via Accuracy, Classification Report (Precision, Recall, F1-Score), and Confusion Matrix.
   * Extracted Feature Importance to highlight key stress indicators.
5. **Serialization**:
   * Serialized the full preprocessing and model pipeline into `student_stress_model.pkl`.

---

## 🖥️ Streamlit Web Application

The application features a clean frontend that enables users to input parameters and receive instant predictions:
* **Input Validation**: Ensures all user inputs fall within expected bounds.
* **Inference Pipeline**: Passes validated data through `student_stress_model.pkl` to compute predictions in real time.

---

## 🛠️ Installation & Local Setup

### Prerequisites
* Python 3.8+
* Git

### Step-by-Step Instructions

1. **Clone the Repository**:
   ```bash
   git clone [https://github.com/Azharmunir96/student-stress-detection-system.git](https://github.com/Azharmunir96/student-stress-detection-system.git)
   cd student-stress-detection-system
