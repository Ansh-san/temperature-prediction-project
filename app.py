import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Global Temperature Predictor", page_icon="🌡️", layout="wide")

@st.cache_resource
def load_artifacts():
    with open("temperature_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

@st.cache_data
def load_data():
    return pd.read_csv("temperature-medie-annuali-1950-2024.csv")

model, le = load_artifacts()
df = load_data()

st.title("🌍 Global Temperature Prediction App")
st.caption("Random Forest model trained on country-level climate data, 1950–2024")

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "🌐 World Globe", "📈 Country Trend"])

#  Predict 
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        country = st.selectbox("Select Country", sorted(le.classes_))
    with col2:
        year = st.number_input("Enter Year", min_value=1950, max_value=2100, value=2025, step=1)

    if st.button("Predict Temperature", type="primary"):
        country_enc = le.transform([country])[0]
        input_data = pd.DataFrame([[country_enc, year]], columns=["country_enc", "year"])
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted mean temperature for **{country}** in **{year}**: **{prediction:.2f} °C**")

        hist = df[df["country"] == country].sort_values("year")
        if not hist.empty:
            latest = hist.iloc[-1]
            st.metric("Most recent recorded value",
                       f"{latest['mean_temperature']:.2f} °C",
                       f"Year {int(latest['year'])}")

        if year > 2024:
            st.caption("⚠️ Years beyond 2024 are extrapolated beyond the training data and less reliable.")

#  World Globe 
with tab2:
    st.subheader("Mean Temperature by Country")
    year_for_globe = st.slider("Year", 1950, 2024, 2024)
    frame = df[df["year"] == year_for_globe]

    fig = go.Figure(data=go.Choropleth(
        locations=frame["iso_code"],
        z=frame["mean_temperature"],
        text=frame["country"],
        colorscale="RdBu_r",
        colorbar_title="°C",
        marker_line_color="white",
        marker_line_width=0.3,
    ))
    fig.update_geos(projection_type="orthographic", showcoastlines=True,
                     showocean=True, oceancolor="LightBlue")
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Drag to rotate the globe. Scroll to zoom.")

#  Country Trend 
with tab3:
    country2 = st.selectbox("Country", sorted(df["country"].unique()), key="trend_country")
    hist = df[df["country"] == country2].sort_values("year")
    fig2 = px.line(hist, x="year", y="mean_temperature", markers=True,
                    title=f"{country2}: Mean Temperature, 1950–2024")
    st.plotly_chart(fig2, use_container_width=True)

st.sidebar.header("About")
st.sidebar.write("Random Forest Regressor (n_estimators=50, max_depth=12)")
st.sidebar.write("Test R² = 0.733 | MAE = 3.15°C | RMSE = 4.24°C")
st.sidebar.write("[GitHub Repo](https://github.com/Ansh-san/temperature-prediction-project)")
