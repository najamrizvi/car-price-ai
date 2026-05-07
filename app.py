import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from predict import predict_price

# ---------------------------
# ⚙️ CONFIG
# ---------------------------
st.set_page_config(
    page_title="🚗 Car Price AI",
    layout="wide",
    page_icon="🚗"
)

DATA_PATH = os.getenv("DATA_PATH", "data/CarPrice_Assignment.csv")
MODEL_PATH = "model.pkl"


# ---------------------------
# 🔐 AUTH STATE
# ---------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


def login():
    st.title("🔐 Car Price AI Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")


if not st.session_state.logged_in:
    login()
    st.stop()


# ---------------------------
# 🌗 THEME
# ---------------------------
theme = st.sidebar.selectbox("🎨 Theme", ["Dark", "Light"])

bg = "#0E1117" if theme == "Dark" else "#F5F7FA"
text = "white" if theme == "Dark" else "black"
card = "rgba(255,255,255,0.08)" if theme == "Dark" else "rgba(0,0,0,0.05)"

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-color: {bg};
    color: {text};
}}

.glass {{
    background: {card};
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.15);
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------
# 📂 DATA LOAD
# ---------------------------
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("Dataset not found")
        st.stop()
    return pd.read_csv(DATA_PATH)


df = load_data()
df_model = df.drop(['car_ID', 'CarName'], axis=1, errors="ignore")

df_encoded = pd.get_dummies(df_model, drop_first=True)

X = df_encoded.drop("price", axis=1)
y = df_encoded["price"]


# ---------------------------
# 🤖 MODEL
# ---------------------------
@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)

    return model


model = get_model()


# ---------------------------
# 🔥 BATCH PREDICTION (FIXED)
# ---------------------------
def batch_predict(model, X, feature_cols):
    X_aligned = X.copy()

    for col in feature_cols:
        if col not in X_aligned.columns:
            X_aligned[col] = 0

    X_aligned = X_aligned[feature_cols]

    return model.predict(X_aligned)


# ---------------------------
# 🧭 NAVIGATION
# ---------------------------
page = st.sidebar.radio("📌 Navigation", ["🏠 Dashboard", "📊 Analytics", "🔮 Prediction"])


# ---------------------------
# 🏠 DASHBOARD
# ---------------------------
if page == "🏠 Dashboard":
    st.title("🚗 Car Price Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.markdown(
        f'<div class="glass"><h4>💰 Avg Price</h4><h2>{df["price"].mean():,.0f}</h2></div>',
        unsafe_allow_html=True
    )

    col2.markdown(
        f'<div class="glass"><h4>🚘 Total Cars</h4><h2>{len(df)}</h2></div>',
        unsafe_allow_html=True
    )

    col3.markdown(
        f'<div class="glass"><h4>⚙️ Avg Horsepower</h4><h2>{df["horsepower"].mean():.0f}</h2></div>',
        unsafe_allow_html=True
    )

    fig = px.histogram(df, x="price", nbins=40, title="Price Distribution")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# 📊 ANALYTICS (FIXED — NO y_pred STATE)
# ---------------------------
elif page == "📊 Analytics":
    st.title("📊 Model Insights")

    try:
        y_pred = batch_predict(model, X, model.feature_names_in_)

        fig = px.scatter(
            x=y,
            y=y_pred,
            labels={"x": "Actual Price", "y": "Predicted Price"},
            title="📈 Actual vs Predicted Prices"
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Analytics error: {str(e)}")


# ---------------------------
# 🔮 PREDICTION (USES predict.py)
# ---------------------------
elif page == "🔮 Prediction":
    st.title("🔮 Predict Car Price")

    with st.form("predict_form"):
        enginesize = st.slider("Engine Size", 50, 500, 150)
        horsepower = st.slider("Horsepower", 50, 300, 100)
        curbweight = st.slider("Weight", 1000, 4000, 2000)

        submit = st.form_submit_button("Predict")

    if submit:
        input_dict = {col: 0 for col in model.feature_names_in_}

        input_dict["enginesize"] = enginesize
        input_dict["horsepower"] = horsepower
        input_dict["curbweight"] = curbweight

        # 🔥 use predict.py here
        prediction = predict_price(input_dict, model.feature_names_in_)

        st.success("Prediction Generated!")

        st.markdown(f"""
        <div class="glass">
            <h3>💰 Estimated Price</h3>
            <h1>{prediction:,.0f}</h1>
        </div>
        """, unsafe_allow_html=True)