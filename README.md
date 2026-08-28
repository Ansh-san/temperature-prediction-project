# 🌍 Global Temperature Atlas

An interactive ML web app predicting country-level annual temperature trends (1950–2024),
built with Python, Scikit-learn, and Streamlit, featuring a rotating 3D world globe.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-3f4f75)
![License](https://img.shields.io/badge/License-MIT-green)

🔗 **Live app:** https://temperature-prediction-project-njuttyubbfvtrq6pvk9jih.streamlit.app/

## Problem & Approach

Climate data is usually explored through static charts. This project turns it into something
interactive: pick any of 196 countries and any year from 1950 to 2100, and get an instant
prediction, backed by a model trained on 75 years of real temperature records, alongside a
rotating globe showing how temperature varies across the world in a given year.

**Approach:**
1. **Data** — 196 countries, annual mean temperature, 1950–2024 (see Dataset below)
2. **Model** — Random Forest Regressor (50 trees, max depth 12) trained on country + year,
   80/20 train-test split (`random_state=42`)
3. **Extrapolation handling** — tree-based models can't predict beyond the range of years they
   were trained on (every year past 2024 would return an identical value). To support future-year
   predictions, the app switches to a per-country linear trend fit for any year beyond 2024
4. **Deployment** — Streamlit app loading the trained model, label encoder, and trend models to
   serve live predictions

## Dataset

[Global Mean Temperature by Country (1950–2024)](https://www.kaggle.com/datasets/lucalullo/global-mean-temperature-by-country-1950-2024) — Kaggle, by Luca Lullo.
196 countries, annual mean surface temperature, 1950–2024.

## Features

- 🔮 Country + year temperature prediction, 1950–2100 (Random Forest for ≤2024, linear trend for >2024)
- 🌐 Interactive rotating world globe (Plotly orthographic projection) colored by mean temperature
- 📈 Per-country historical trend chart, 1950–2024

## Model Evaluation

Overall test-set performance: **R² = 0.733, MAE = 3.15°C, RMSE = 4.24°C**

Performance varies notably by country. Prediction error is lowest for smaller, climatically
uniform countries and highest for large, climatically diverse ones — a single national average
hides significant internal variation for a country like Canada or Kazakhstan, making its
year-to-year mean harder to predict than a smaller country like Namibia or Uruguay.

| Best predicted (MAE) | Worst predicted (MAE) |
|---|---|
| Namibia — 0.23°C | Canada — 18.13°C |
| Uruguay — 0.25°C | Estonia — 11.85°C |
| Mexico — 0.27°C | D.P.R. of Korea — 11.62°C |
| Zimbabwe — 0.34°C | Kyrgyz Republic — 10.71°C |
| Madagascar — 0.35°C | Czech Republic — 10.14°C |

## Screenshots

| Predict | World Globe | Country Trend |
|---|---|---|
| ![Predict](screenshots/TEMPPREDICT.png) | ![Globe](screenshots/Globe.png) | ![Trend](screenshots/trend.png) |

## Project Structure
├── app.py # Streamlit web app
├── temperature_model (1).pkl # trained Random Forest model
├── label_encoder.pkl # country label encoder
├── trend_models.pkl # per-country linear trend coefficients
├── temperature-medie-annuali-1950-2024.csv # dataset
├── screenshots/ # app screenshots
├── requirements.txt
├── LICENSE
└── README.md


## How to Run Locally

```bash
git clone https://github.com/Ansh-san/temperature-prediction-project.git
cd temperature-prediction-project
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Limitations & Future Work

- The model predicts temperature from country and year only — it does not use actual historical
  temperature values, precipitation, CO₂ levels, or other causal climate factors, so it captures
  trend patterns per country rather than true climate dynamics.
- No cross-validation or hyperparameter search was performed beyond manual tuning of tree count and depth.
- Years beyond 2024 use a linear trend fit rather than the Random Forest model, since tree-based
  models cannot extrapolate past their training range — this is a rough projection, not a forecast.
- Future work could add confidence intervals on predictions, incorporate additional climate
  covariates, or perform proper hyperparameter tuning via cross-validation.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
