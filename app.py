import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Global Temperature Predictor", page_icon="🌡️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #0A0E1A; }

.atlas-hero { padding: 1.2rem 0 0.6rem 0; border-bottom: 1px solid #2A3350; margin-bottom: 1.4rem; }
.atlas-hero h1 { font-family: 'Fraunces', serif; font-weight: 600; font-size: 2.3rem; color: #EDEFF5; margin-bottom: 0.2rem; letter-spacing: -0.01em; }
.atlas-hero p { color: #8993B0; font-size: 0.95rem; margin: 0; }

.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: none; }
.stTabs [data-baseweb="tab"] {
    background: #131A2B; border: 1px solid #2A3350; border-radius: 999px;
    padding: 6px 18px; color: #8993B0; font-weight: 500;
}
.stTabs [aria-selected="true"] { background: #1C2541; color: #EDEFF5; border-color: #FF8A4C; }

.stButton > button {
    background: linear-gradient(135deg, #4FB8E8, #FF8A4C);
    color: #0A0E1A; border: none; border-radius: 8px; font-weight: 600;
    padding: 0.5rem 1.4rem; transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 14px rgba(255,138,76,0.35); }

.instrument-card {
    background: #131A2B; border: 1px solid #2A3350; border-radius: 14px;
    padding: 1.4rem 1.6rem; margin-top: 0.8rem; animation: fadeIn 0.4s ease;
}
.instrument-card .reading { font-family: 'IBM Plex Mono', monospace; font-size: 2.6rem; font-weight: 600; color: #EDEFF5; }
.instrument-card .reading span { font-size: 1.2rem; color: #8993B0; }
.instrument-card .label { color: #8993B0; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.3rem; }
.instrument-card .method { font-size: 0.78rem; color: #4FB8E8; margin-top: 0.4rem; font-family: 'IBM Plex Mono'; }

.gauge-track {
    position: relative; height: 10px; border-radius: 6px; margin: 1rem 0 0.4rem 0;
    background: linear-gradient(90deg, #4FB8E8 0%, #8993B0 45%, #FF8A4C 75%, #FF4757 100%);
}
.gauge-marker {
    position: absolute; top: -7px; width: 3px; height: 24px; background: #EDEFF5;
    border-radius: 2px; box-shadow: 0 0 8px rgba(237,239,245,0.8);
}
.gauge-labels { display: flex; justify-content: space-between; font-family: 'IBM Plex Mono'; font-size: 0.7rem; color: #8993B0; }

@keyframes fadeIn { from { opacity: 0; transform: translateY(6px);} to { opacity: 1; transform: translateY(0);} }
</style>

<div class="atlas-hero">
<h1>Global Temperature Atlas</h1>
<p>Country-level climate trends, 1950–2024 — Random Forest + linear trend model</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    with open("temperature_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("trend_models.pkl", "rb") as f:
        trend_models = pickle.load(f)
    return model, le, trend_models

@st.cache_data
def load_data():
    return pd.read_csv("temperature-medie-annuali-1950-2024.csv")

model, le, trend_models = load_artifacts()
df = load_data()

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "🌐 World Globe", "📈 Country Trend"])

# ---------------- Predict ----------------
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        country = st.selectbox("Select Country", sorted(le.classes_))
    with col2:
        year = st.number_input("Enter Year", min_value=1950, max_value=2100, value=2025, step=1)

    if st.button("Predict Temperature", type="primary"):
        if year <= 2024:
            country_enc = le.transform([country])[0]
            input_data = pd.DataFrame([[country_enc, year]], columns=["country_enc", "year"])
            prediction = model.predict(input_data)[0]
            method = "RANDOM FOREST · INTERPOLATED"
        else:
            slope, intercept = trend_models[country]
            prediction = slope * year + intercept
            method = "LINEAR TREND · EXTRAPOLATED"

        gauge_min, gauge_max = -30, 45
        pct = max(0, min(100, (prediction - gauge_min) / (gauge_max - gauge_min) * 100))

        hist = df[df["country"] == country].sort_values("year")
        delta_html = ""
        if not hist.empty:
            latest = hist.iloc[-1]
            delta_html = (f'<div class="label" style="margin-top:0.8rem;">VS MOST RECENT RECORDED ({int(latest["year"])})</div>'
                           f'<div style="font-family:\'IBM Plex Mono\'; color:#EDEFF5;">{latest["mean_temperature"]:.2f} °C</div>')

        st.markdown(f"""
        <div class="instrument-card">
            <div class="label">{country.upper()} · {year}</div>
            <div class="reading">{prediction:.2f}<span> °C</span></div>
            <div class="method">{method}</div>
            <div class="gauge-track"><div class="gauge-marker" style="left: calc({pct}% - 1.5px);"></div></div>
            <div class="gauge-labels"><span>{gauge_min}°C</span><span>{gauge_max}°C</span></div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)

        if year > 2024:
            st.caption("⚠️ This is a linear trend projection, not a Random Forest prediction — treat it as a rough estimate, not a forecast.")

# ---------------- World Globe ----------------
with tab2:
    st.subheader("Mean Temperature by Country")
    year_for_globe = st.slider("Year", 1950, 2024, 2024)
    frame = df[df["year"] == year_for_globe]

    fig = go.Figure(data=go.Choropleth(
        locations=frame["iso_code"],
        z=frame["mean_temperature"],
        text=frame["country"],
        colorscale=[[0, "#4FB8E8"], [0.5, "#8993B0"], [0.75, "#FF8A4C"], [1, "#FF4757"]],
        colorbar_title="°C",
        marker_line_color="white",
        marker_line_width=0.3,
    ))
    fig.update_geos(projection_type="orthographic", showcoastlines=True,
                     showocean=True, oceancolor="#0A0E1A")
    fig.update_layout(
        height=600, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#0A0E1A",
        font=dict(family="IBM Plex Mono", color="#EDEFF5"),
        geo=dict(bgcolor="#0A0E1A", landcolor="#131A2B", oceancolor="#0A0E1A", lakecolor="#0A0E1A"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Drag to rotate the globe. Scroll to zoom.")

# ---------------- Country Trend ----------------
with tab3:
    country2 = st.selectbox("Country", sorted(df["country"].unique()), key="trend_country")
    hist = df[df["country"] == country2].sort_values("year")
    fig2 = px.line(hist, x="year", y="mean_temperature", markers=True,
                    title=f"{country2}: Mean Temperature, 1950–2024")
    fig2.update_traces(line_color="#FF8A4C")
    fig2.update_layout(paper_bgcolor="#0A0E1A", plot_bgcolor="#131A2B", font=dict(family="IBM Plex Mono", color="#EDEFF5"))
    st.plotly_chart(fig2, use_container_width=True)

st.sidebar.header("About")
st.sidebar.write("Random Forest Regressor (n_estimators=50, max_depth=12)")
st.sidebar.write("Test R² = 0.733 | MAE = 3.15°C | RMSE = 4.24°C")
st.sidebar.write("[GitHub Repo](https://github.com/Ansh-san/temperature-prediction-project)")
