# Consumer Behavior Classification: Online vs Store Shopping

## Overview

This project focuses on predicting consumer shopping behavior (Online, Store, Hybrid) using a structured dataset of demographic and behavioral features.

During exploratory data analysis (EDA), a strong class imbalance was identified, with the majority of samples belonging to the "Store" category. To address this issue, multiple modeling strategies were explored, including class weighting and data resampling techniques.

Four different approaches were evaluated:
- Logistic Regression (baseline)
- Logistic Regression with SMOTE
- Support Vector Machine (SVM)
- Random Forest

The results demonstrate that handling class imbalance significantly improves performance, particularly for minority classes.

---

## Installation & Setup
🪟Windows:
```bash
git clone https://github.com/soha-kd/consumer_behavior_project.git
cd consumer_behavior_project
python -m venv test_env
test_env\Scripts\activate
python -m pip install --upgrade pip
pip install .
python -m consumer_behavior_project

```
🐧Linux/🍏MacOS:
```bash
git clone https://github.com/soha-kd/consumer_behavior_project.git
cd consumer_behavior_project
python3 -m venv test_env
source test_env/bin/activate
python3 -m pip install --upgrade pip
pip install .
python3 -m consumer_behavior_project
```

## Project Structure

```id="k3f9qs"
consumer_behavior_project/
│── consumer_behavior_project/
│   │── __init__.py          # Package initialization
│   │── __main__.py          # Entry point to run the project
│   │── model.py             # Model training logic
│   │── preprocessing.py     # Data preprocessing pipeline
│   │── evaluation.py        # Model evaluation (metrics, plots)
│   │── utils.py             # Helper functions
│
│── data/                    # Dataset used for training and testing
│
│── notebooks/
│   │── exploration.ipynb    # Exploratory Data Analysis (EDA) with inline visualizations
│   │── modeling_steps.ipynb # Step-by-step modeling process and results visualization
│
│── outputs/
│   │── plots/               # Saved plots (confusion matrix, ROC, etc.)
│
│── pyproject.toml           # Project configuration and dependencies
│── README.md                # Project documentation
```

### Notebooks

* **exploration.ipynb**: Focuses on data exploration and understanding feature distributions, correlations, and class imbalance.
* **modeling_steps.ipynb**: Demonstrates the full modeling workflow, including preprocessing, training, and evaluation, with inline results and visualizations.


### Notebooks

* **exploration.ipynb**: Focuses on data exploration and understanding feature distributions, correlations, and class imbalance.
* **modeling_steps.ipynb**: Demonstrates the full modeling workflow, including preprocessing, training, and evaluation, with inline results and visualizations.

```

## Methodology

### Data Exploration

The dataset contains demographic, behavioral, and preference-based features related to consumer shopping habits.

EDA showed:
- No missing values
- Mostly numerical features
- A strong class imbalance:
  - Store (majority)
  - Online and Hybrid (minority)

---

### Preprocessing

The pipeline includes:
- Encoding categorical variables
- Feature scaling
- Train-test split
- Label encoding

All steps were applied carefully to avoid data leakage.

---

### Models

Four models were trained and compared:

- **Logistic Regression**  
  Baseline model with solid overall performance but weaker on minority classes.

- **Logistic Regression (SMOTE)**  
  Uses oversampling to balance the data. Achieved the best performance, especially on minority classes.

- **SVM**  
  Captures non-linear patterns but is sensitive to class imbalance.

- **Random Forest**  
  Strong and interpretable model with feature importance, but slightly lower performance than SMOTE approach.

---

### Evaluation

Models were evaluated using:
- Accuracy
- F1 Score

Additional visualizations:
- Confusion matrices
- ROC curves
- Precision-Recall curves

---

### 📊 Visualization

- Plots are displayed inline in the notebook for easy analysis.
- All generated plots are also automatically saved in: outputs/plots/


---

### Results & Conclusion

The best model was **Logistic Regression with SMOTE**, showing improved balance across all classes.
