import streamlit as st

st.set_page_config(
    page_title="NIFTY 50 Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Sidebar navigation ──────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/NSE_logo.svg/320px-NSE_logo.svg.png",
    width=160,
)
st.sidebar.title("NIFTY 50 Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📅 Yearly Prediction", "🗓️ Monthly Prediction", "ℹ️ About"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Built by Aparna · ML Portfolio Project")

# ── Page routing ─────────────────────────────────────────────────────────────
if page == "🏠 Home":
    from pages import home
    home.show()

elif page == "📅 Yearly Prediction":
    from pages import yearly
    yearly.show()

elif page == "🗓️ Monthly Prediction":
    from pages import monthly
    monthly.show()

elif page == "ℹ️ About":
    from pages import about
    about.show()
