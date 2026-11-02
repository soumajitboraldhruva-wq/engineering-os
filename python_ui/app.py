import requests
import streamlit as st

st.set_page_config(page_title="ENG-OS Quantitative Analysis", layout="wide")
st.title("ENG-OS — Quantitative Analysis")
st.caption("Python UI → FastAPI → deterministic quantitative engine")

api = st.sidebar.text_input("FastAPI URL", "http://127.0.0.1:8000")
mode = st.sidebar.selectbox("Mode", ["Sequence / Forecast"])
horizon = st.sidebar.number_input("Forecast horizon", 1, 100, 5)

raw = st.text_area(
    "Enter sequence values",
    "1, 4, 9, 16, 25",
    height=120,
)
values=[]
try:
    values=[float(v.strip()) for v in raw.replace("\n", ",").split(",") if v.strip()]
except ValueError:
    st.error("Use comma/newline separated numeric values.")

if st.button("Analyze", type="primary"):
    if len(values)<3:
        st.error("Enter at least 3 values.")
    else:
        try:
            r=requests.post(
                f"{api.rstrip('/')}/api/quant/analyze",
                json={"values":values, "horizon":int(horizon)},
                timeout=60,
            )
            r.raise_for_status()
            data=r.json()
            c1,c2,c3=st.columns(3)
            c1.metric("Category", data.get("category","—"))
            c2.metric("Next value", data.get("next_value","—"))
            c3.metric("Evidence", f"{data.get('evidence_score',0):.1f}")
            st.subheader("Formula")
            st.code(data.get("formula","—"))
            st.subheader("Logic")
            st.write(data.get("logic",""))
            if data.get("predictions"):
                st.subheader("Forecast")
                st.line_chart(data["predictions"])
            if data.get("warning"):
                st.warning(data["warning"])
            with st.expander("Raw API result"):
                st.json(data)
        except Exception as exc:
            st.error(f"API request failed: {exc}")
