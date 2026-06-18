import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load data

matches = pd.read_csv("data/matches.csv")
deliveries = pd.read_csv("data/deliveries.csv")

print("Data loaded")


# -------------------------------
# Create first innings totals
# -------------------------------

first_innings = deliveries[
    deliveries["innings"] == 1
].groupby(
    "match_id"
)["total_runs"].sum()


# -------------------------------
# Second innings data
# -------------------------------

chase = deliveries[
    deliveries["innings"] == 2
].copy()


print("Second innings rows:", len(chase))


# Current score

chase["current_score"] = chase.groupby(
    "match_id"
)["total_runs"].cumsum()


# Balls

chase["balls"] = chase.groupby(
    "match_id"
).cumcount() + 1


# Wickets

chase["is_wicket"] = chase["is_wicket"].astype(int)

chase["wickets_lost"] = chase.groupby(
    "match_id"
)["is_wicket"].cumsum()



# Target

chase["target"] = chase["match_id"].map(first_innings) + 1



# Features

chase["runs_left"] = (
    chase["target"] -
    chase["current_score"]
)


chase["balls_left"] = (
    120 -
    chase["balls"]
)


chase["wickets_left"] = (
    10 -
    chase["wickets_lost"]
)


chase["current_run_rate"] = (
    chase["current_score"] /
    (chase["balls"]/6)
)


chase["required_run_rate"] = (
    chase["runs_left"] /
    (chase["balls_left"]/6)
)



# Create result using match winner

winner = matches[
["match_id","winner"]
]


chase = chase.merge(
    winner,
    on="match_id",
    how="left"
)


# remove missing winners

chase = chase.dropna(
    subset=["winner"]
)



print("After winner merge:", len(chase))



# Result

chase["result"] = (
    chase["batting_team"] ==
    chase["winner"]
).astype(int)



# Clean
chase = chase.replace(
    [float("inf"),-float("inf")],
    0
)

chase = chase.fillna(0)



print("Training rows:",len(chase))


# Model data

X = chase[
[
"runs_left",
"balls_left",
"wickets_left",
"current_run_rate",
"required_run_rate"
]
]


y = chase["result"]



# Split

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# Train

model = LogisticRegression()

model.fit(
    X_train,
    y_train
)



# Accuracy

pred = model.predict(X_test)

print(
"Accuracy:",
accuracy_score(
y_test,
pred
)
)



# Test prediction
sample = pd.DataFrame(
[
[50,30,6,8.5,10]
],
columns=[
"runs_left",
"balls_left",
"wickets_left",
"current_run_rate",
"required_run_rate"
]
)

print(
"Prediction probability:",
model.predict_proba(sample)
)
import joblib

joblib.dump(
    model,
    "model/cricket_model.pkl"
)

print("Model saved")