import streamlit as st
import joblib
import pandas as pd


model = joblib.load(
    "model/cricket_model.pkl"
)


st.title("🏏 Cricket Win Predictor")

st.write("AI based cricket match winning probability")


runs_left = st.number_input(
    "Runs Left",
    min_value=0
)

balls_left = st.number_input(
    "Balls Left",
    min_value=0
)

wickets_left = st.number_input(
    "Wickets Left",
    min_value=0,
    max_value=10
)

current_rr = st.number_input(
    "Current Run Rate",
    min_value=0.0
)

required_rr = st.number_input(
    "Required Run Rate",
    min_value=0.0
)



if st.button("Predict Winner"):

    data = pd.DataFrame(
        [[
            runs_left,
            balls_left,
            wickets_left,
            current_rr,
            required_rr
        ]],
        columns=[
            "runs_left",
            "balls_left",
            "wickets_left",
            "current_run_rate",
            "required_run_rate"
        ]
    )


    prediction = model.predict_proba(data)


    win = prediction[0][1] * 100
    lose = prediction[0][0] * 100


    if required_rr > 20:
        win = min(win,10)
        lose = 100 - win


    if balls_left < 18 and runs_left > 50:
        win = min(win,5)
        lose = 100 - win


    st.success(
        f"Batting Team Win Chance: {win:.2f}%"
    )

    st.info(
        f"Opposition Win Chance: {lose:.2f}%"
    )