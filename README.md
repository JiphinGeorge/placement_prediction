# 🎓 Student Placement Prediction System

![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

> A Streamlit‑based machine learning application that predicts whether a student will be placed in campus recruitment, using academic, work experience and employability indicators.


## 📌 What the project does

The repository contains code to train an XGBoost classifier on the [Kaggle Campus Recruitment Dataset](https://www.kaggle.com/datasets/), perform feature engineering, and serve the resulting model via an interactive web app. Users can input student details and receive a placement verdict along with confidence scores.


## 🚀 Why it's useful

- **Predictive insights** for educators and students about placement likelihood
- Demonstrates end‑to‑end ML workflow (data processing → training → deployment)
- Lightweight and self‑contained; no external services required
- A handy portfolio project for machine learning practitioners


## 📂 Project structure

```
placement_prediction/
│
├── data/                    # raw dataset (CSV)
│   └── placement.csv
│
├── model/                   # serialized artifacts
│   ├── placement_model.pkl
│   └── label_encoders.pkl
│
├── app.py                   # Streamlit application
├── train_model.py           # training & hyperparameter tuning script
├── README.md                # this file
└── .github/ ...             # ancillary project metadata
```


## 🛠 Getting started

### Prerequisites

- Python 3.8 or higher
- `pip` for installing dependencies

### Installation

Clone the repository and install required packages:

```bash
git clone <repo-url> placement_prediction
cd placement_prediction
pip install pandas numpy scikit-learn xgboost streamlit joblib
```

(The model in `model/` is already trained; re‑training is optional.)

### Running the app

1. **(Optional)** Train or retrain the model:
   ```bash
   python train_model.py
   ```
   This will read `data/placement.csv`, perform encoding & feature engineering, run a `GridSearchCV` for hyperparameter tuning, and save the best model and encoders under `model/`.

2. **Start the Streamlit interface**:
   ```bash
   python -m streamlit run app.py
   ```

3. Open your browser at [http://localhost:8501](http://localhost:8501).


## ⚙️ Usage example

Once the app is running, use the sidebar to supply the following inputs:

- Gender, SSC/HSC/Degree/MBA percentages
- Work experience (Yes/No)
- Employability test score
- MBA specialisation

Click **Predict Placement** to view the result and probability.


## 📊 Data & features

The model uses the following predictors:

- `gender`, `ssc_p`, `hsc_p`, `degree_p`, `mba_p`
- `workex`, `etest_p`, `specialisation`
- Engineered features: `academic_avg`, `employability_score`

Target variable: `status` (placed/not placed).


## 📚 Help & documentation

For detailed setup and contribution guidelines, refer to the project’s [CONTRIBUTING.md](docs/CONTRIBUTING.md) (when available). Questions or issues can be opened via GitHub Issues.


## 🤝 Contributing

Contributions are welcome! Please fork the repo and submit a pull request. See `docs/CONTRIBUTING.md` for the full guidelines.


## 🧑‍💼 Maintainers

- **Jiphin George** – original author and current maintainer


## 📄 License

This project is available under the [MIT License](LICENSE).


---

*Last updated:* February 19, 2026
