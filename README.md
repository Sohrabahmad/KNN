# ❤️ Heart Disease Prediction using K-Nearest Neighbors (KNN)

This project implements a **Heart Disease Prediction system using the K-Nearest Neighbors (KNN) classification algorithm**.

The model uses patient medical attributes to predict whether a patient is likely to have heart disease. Different values of **K** are tested, followed by **GridSearchCV** for hyperparameter tuning. Feature scaling is performed using `StandardScaler` because KNN is distance-based.

---

## 📌 Project Overview

The objective of this project is to build a machine learning classification model that predicts heart disease based on clinical and medical features.

The project covers:

- Data loading and preprocessing
- Feature and target separation
- Train-test splitting
- Feature standardization
- KNN classification
- Testing different K values
- Model evaluation
- Hyperparameter tuning using `GridSearchCV`
- Building a preprocessing and KNN pipeline

---

## 📂 Dataset

The dataset contains **303 records** and **14 columns**.

### Features

| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Sex of the patient |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar |
| `restecg` | Resting electrocardiographic results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels |
| `thal` | Thalassemia |
| `target` | Heart disease target variable |

### Dataset Shape

```text
Rows: 303
Columns: 14
```

---

## 🧠 Algorithm Used

### K-Nearest Neighbors (KNN)

KNN is a supervised machine learning algorithm used for classification and regression.

For classification, KNN:

1. Calculates the distance between data points.
2. Finds the nearest `K` observations.
3. Checks the classes of those neighbors.
4. Assigns the majority class to the new observation.

Since KNN relies on distance calculations, **feature scaling is important**.

---

## ⚙️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Jupyter Notebook
- StandardScaler
- KNeighborsClassifier
- GridSearchCV
- Pipeline

---

## 🔄 Machine Learning Workflow

```text
Load Dataset
     ↓
Separate Features and Target
     ↓
Train-Test Split
     ↓
StandardScaler
     ↓
Train KNN Model
     ↓
Test Different K Values
     ↓
Evaluate Model
     ↓
GridSearchCV
     ↓
Select Best K
     ↓
Final KNN Model
```

---

## 📊 Train-Test Split

The dataset is divided using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This results in:

- **80% training data**
- **20% testing data**

---

## 📏 Feature Scaling

`StandardScaler` is used before applying KNN:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

This standardizes the features so that variables with larger numerical ranges do not dominate the distance calculation.

---

## 🔬 KNN Model Testing

The notebook tests three K values:

- K = 3
- K = 5
- K = 7

### Results

| K | Accuracy | Precision | Recall | F1 Score |
|---:|---:|---:|---:|---:|
| 3 | 85.25% | 92.59% | 78.13% | 84.75% |
| 5 | 90.16% | 93.33% | 87.50% | 90.32% |
| **7** | **91.80%** | **93.55%** | **90.63%** | **92.06%** |

Based on the tested values, **K = 7** produced the best performance.

---

## 🎯 Hyperparameter Tuning

`GridSearchCV` is used to find the optimal number of neighbors.

Tested values:

```python
param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11]
}
```

The search uses:

```python
cv=5
scoring="recall"
```

### Best Parameter

```text
n_neighbors = 7
```

The tuned model achieved:

- **Accuracy:** 91.80%
- **Precision:** 93.55%
- **Recall:** 90.63%
- **F1 Score:** 92.06%

---

## 🔗 Pipeline Implementation

The project also implements a Scikit-learn pipeline combining feature scaling and KNN:

```python
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])
```

GridSearchCV is then applied to the pipeline:

```python
param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11]
}
```

The best configuration was:

```text
knn__n_neighbors = 7
```

---

## 📈 Final Model Performance

The final tuned KNN model achieved:

| Metric | Score |
|---|---:|
| Accuracy | **91.80%** |
| Precision | **93.55%** |
| Recall | **90.63%** |
| F1 Score | **92.06%** |

These results are based on the 20% held-out test set using `random_state=42`.

---

## 📁 Project Structure

```text
Heart-Disease-KNN/
│
├── knn.ipynb
├── heart(3).csv
└── README.md
```

> **Note:** The notebook currently loads the dataset using `pd.read_csv("heart.csv")`. If the dataset is stored with the uploaded filename `heart(3).csv`, update the notebook to:
>
> ```python
> data_df = pd.read_csv("heart(3).csv")
> ```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Heart-Disease-KNN
```

### 2. Install Dependencies

```bash
pip install pandas scikit-learn jupyter
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open

```text
knn.ipynb
```

Run the notebook cells sequentially.

---

## 📦 Required Libraries

```python
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
```

---

## 🎯 Learning Outcomes

Through this project, the following concepts are demonstrated:

- Understanding KNN classification
- Importance of feature scaling
- Train-test splitting
- Model evaluation
- Accuracy, precision, recall and F1 score
- Hyperparameter tuning
- Cross-validation
- GridSearchCV
- Scikit-learn pipelines
- Selecting the optimal value of K

---

## ⚠️ Disclaimer

This project is developed for **educational and machine learning practice purposes**. The predictions should not be considered medical advice or a substitute for professional medical diagnosis.

---

## 👨‍💻 Author

**Sohrab Ahmad Siddiqui**

B.Tech Computer Science Engineering (AI & ML)

---

## ⭐ Project Highlights

- 🫀 Heart disease classification
- 🤖 K-Nearest Neighbors algorithm
- 📏 StandardScaler preprocessing
- 🔍 GridSearchCV hyperparameter tuning
- 🔗 Scikit-learn Pipeline
- 📊 **91.80% test accuracy**
- 🎯 **90.63% recall**
- ⭐ Best K = **7**