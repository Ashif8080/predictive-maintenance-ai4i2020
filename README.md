# 🔧 Predictive Maintenance – Machine Failure Prediction (AI4I 2020)

A beginner-friendly data science project that explores the **AI4I 2020 Predictive Maintenance Dataset** and builds a **Random Forest** model to predict machine failures from sensor readings.

> 📌 **Note:** This is a learning project. Feedback and suggestions are very welcome!

---

## 📂 Project Structure

```
predictive-maintenance-ai4i2020/
├── AI4I_2020.ipynb     # Main notebook (EDA + modeling)
├── sample.csv          # First 5 rows of the dataset (for preview)
└── README.md
```

---

## 📥 Dataset Source

- **Source:** Kaggle Datasets
- **Link:** [Predictive Maintenance Dataset (AI4I 2020)](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)

Download `ai4i2020.csv` from Kaggle and place it in the project folder (update the path in the notebook if needed).

---

## 📊 Dataset Overview

- **Rows:** 10,000 | **Columns:** 14 | **Missing values:** None
- **Machine types:** L (6,000), M (2,997), H (1,003)
- **Total machine failures:** 339 (~3.4%) → a heavily **imbalanced** dataset

| Feature | Description |
|---|---|
| Air temperature [K] | Ambient air temperature |
| Process temperature [K] | Temperature during the process |
| Rotational speed [rpm] | Spindle speed |
| Torque [Nm] | Torque applied |
| Tool wear [min] | Cumulative tool usage time |
| Machine failure | Target (0 = OK, 1 = Failure) |
| TWF / HDF / PWF / OSF / RNF | Specific failure modes (Tool wear, Heat dissipation, Power, Overstrain, Random) |

---

## 🔍 Exploratory Data Analysis

- Statistical summary of sensor features
- Failure counts by machine type and by failure mode
- Total tool wear time: ~17,992 hours (~750 days)
- Visualizations (Matplotlib):
  - Torque vs. RPM scatter plot
  - Failure mode distribution (bar chart)
  - Failures by machine type (pie chart)
  - Machine type share of production (pie chart)

**Failures by type:** L = 235 · M = 83 · H = 21

**Failures by mode:** HDF = 115 · OSF = 98 · PWF = 95 · TWF = 46 · RNF = 19

---

## 🤖 Modeling

- **Features:** Process temperature, Rotational speed, Torque
- **Target:** Machine failure
- **Split:** 80/20 train-test, stratified (`random_state=32`)
- **Model:** `RandomForestClassifier` with `class_weight={0:1, 1:10}` to handle imbalance

### Results

| Setting | Precision (failure) | Recall (failure) | F1 (failure) | Accuracy |
|---|---|---|---|---|
| Default threshold (0.50) | 0.87 | 0.29 | 0.44 | 0.97 |
| Tuned threshold (0.20) | 0.30 | 0.43 | 0.35 | 0.95 |

**Key insight:** Accuracy alone is misleading on imbalanced data. The default model looked great (97%) but missed most real failures. Lowering the decision threshold improved **recall** at the cost of **precision**, which is a typical trade-off in predictive maintenance, where missing a failure is usually costlier than a false alarm.

---

## 🛠️ Prediction Function

A simple helper checks a machine's risk from live sensor values:

```python
check_machine_failure(process_temp=309.1, rpm=7380, torque=52.5)
```

```
Probability of failure: 29.00%
Warning: Machine has potential failure risk!
Request for immediate checkup
```

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install numpy pandas matplotlib scikit-learn jupyter

# 2. Download ai4i2020.csv from Kaggle, put it in the folder, then launch
jupyter notebook AI4I_2020.ipynb
```

---

## 🧰 Tech Stack

Python · Pandas · NumPy · Matplotlib · Scikit-learn

---

## 🔮 Future Improvements

- Try SMOTE / undersampling for class imbalance
- Compare models (XGBoost, Logistic Regression, SVM)
- Add feature engineering (e.g., temperature difference, power = torque × speed)
- Use cross-validation, ROC-AUC and Precision-Recall curves
- Include more features (Tool wear, Type) in the model
- Deploy as a small Streamlit app

---

## 🙏 Acknowledgements

Dataset: [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020) on Kaggle.

---

## 📬 Feedback

This is part of my learning journey. If you spot mistakes or have suggestions, please open an issue or leave a comment. Thank you! 😊
