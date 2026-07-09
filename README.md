# Temperature Trend Prediction 🌡️

A machine learning project for predicting annual temperature trends using historical climate data,
built with Python, Pandas, Scikit-learn, and Streamlit — deployed as an interactive web app.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

## Overview

This project builds a machine learning model to predict annual average temperature trends
from historical climate records, and serves predictions through an interactive Streamlit
web application.

## Dataset

- **Source:** `<ADD DATASET SOURCE — e.g. Kaggle link, ISTAT (Italian national statistics), or original data provider>`
- **Region/Coverage:** `<ADD — the filename suggests Italian annual average temperature data; confirm region/country>`
- **Time range:** 1950–2024
- **File:** `temperature-medie-annuali-1950-2024.csv`

## Approach

1. **Data preprocessing** — `<describe cleaning, handling missing years/values, encoding categorical features via label_encoder.pkl>`
2. **Feature engineering** — `<e.g. lag features, rolling averages, year/region encoding>`
3. **Model** — `<e.g. Linear Regression / Random Forest / etc. — name the actual algorithm used in temperature_model.pkl>`
4. **Deployment** — Streamlit app (`app.py`) loads the trained model and label encoder to serve live predictions

## Results

| Metric | Value |
|---|---|
| R² Score | `<X>` |
| MAE | `<X>` |
| RMSE | `<X>` |

`<Add a plot here if you have one — e.g. predicted vs actual temperature over years>`

![Predicted vs Actual](images/predicted_vs_actual.png)

## App Screenshot

`<Add a screenshot of your running Streamlit app here>`

![App Screenshot](images/app_screenshot.png)

## Project Structure

```
├── app.py                                    # Streamlit web app
├── temperature_model.pkl                     # trained ML model
├── label_encoder.pkl                         # encoder for categorical features
├── temperature-medie-annuali-1950-2024.csv   # dataset
├── requirements.txt
├── LICENSE
└── README.md
```

## How to Run

```bash
git clone https://github.com/Ansh-san/temperature-prediction-project.git
cd temperature-prediction-project
pip install -r requirements.txt
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

## Limitations & Future Work

- `<e.g. Model trained on a single region's data — may not generalize to other climates>`
- `<e.g. No cross-validation performed / limited hyperparameter tuning>`
- `<e.g. Could add confidence intervals to predictions, or extend to monthly-level forecasting>`

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Author
Ansh Parashar
