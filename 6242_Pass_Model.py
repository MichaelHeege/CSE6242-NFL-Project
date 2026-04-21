import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import joblib
#import seaborn as sns

#data import from nflfastr from 2010 to 2020 seasons
pbp_cols = [
    "play_id",
    "game_id",
    "season",
    "season_type",
    "week",
    "posteam",
    "defteam",
    "pass_attempt",
    "complete_pass",
    "air_yards",
    "pass_location",
    "yardline_100",
    "ydstogo",
    "down",
    "qtr",
    "game_seconds_remaining",
    "shotgun",
    "qb_spike",
    "qb_kneel",
    "play_deleted",
    "penalty",
    "yards_after_catch",
    "yards_gained",
]

import os

def load_data(data_path="data", start_year=2010, end_year=2021):
    dfs = []

    for year in range(start_year, end_year):
        file_path = os.path.join(data_path, f"play_by_play_{year}.csv")
        df_year = pd.read_csv(file_path, usecols=pbp_cols, low_memory=False)
        dfs.append(df_year)

    return pd.concat(dfs, ignore_index=True)


#pkl file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "pass_model.pkl")
FEATURES_FILE = os.path.join(BASE_DIR, "feature_cols.pkl")
OFF_RANK_FILE = os.path.join(BASE_DIR, "off_rank_lookup.pkl")
DEF_RANK_FILE = os.path.join(BASE_DIR, "def_rank_lookup.pkl")

def clean_data(df):
    df = df.copy()

    # Keep real pass attempts with known air yards
    df = df[(df["pass_attempt"] == 1) & (df["air_yards"].notna())]

    # Remove non-representative plays
    df = df[(df["qb_spike"] == 0) & (df["qb_kneel"] == 0)]
    df = df[df["play_deleted"] == 0]
    df = df[df["penalty"] == 0]

    # Clean target
    df = df[df["complete_pass"].isin([0, 1])]

    # Clean feature ranges
    df = df[(df["air_yards"] > -20) & (df["air_yards"] < 80)]
    df = df[(df["yardline_100"] >= 0) & (df["yardline_100"] <= 100)]
    df = df[(df["ydstogo"] >= 1) & (df["ydstogo"] <= 30)]
    df = df[df["down"].isin([1, 2, 3, 4])]
    df = df[df["qtr"].isin([1, 2, 3, 4])]
    df = df[df["pass_location"].isin(["left", "middle", "right"])]

    df["shotgun"] = df["shotgun"].fillna(0).astype(int)
    df["complete_pass"] = df["complete_pass"].astype(int)
    df["down"] = df["down"].astype(int)
    df["qtr"] = df["qtr"].astype(int)
    df["season"] = df["season"].astype(int)

    return df

#add in offense and defense rank (from Michael Heege's code with some modifications)
def add_team_ranks(df):
    df = df.copy()

    off_rank_frames = []
    def_rank_frames = []

    for season in sorted(df["season"].unique()):
        season_reg = df[(df["season"] == season) & (df["season_type"] == "REG")].copy()

        off = (
            season_reg.groupby("posteam", as_index=False)["complete_pass"]
            .mean()
            .rename(columns={"complete_pass": "off_completion_pct"})
        )
        off["off_rank"] = off["off_completion_pct"].rank(method="min", ascending=False).astype(int)
        off["season"] = season
        off = off.rename(columns={"posteam": "team"})
        off_rank_frames.append(off[["season", "team", "off_rank"]])

        defense = (
            season_reg.groupby("defteam", as_index=False)["complete_pass"]
            .mean()
            .rename(columns={"complete_pass": "def_completion_pct_allowed"})
        )
        defense["def_rank"] = defense["def_completion_pct_allowed"].rank(method="min", ascending=True).astype(int)
        defense["season"] = season
        defense = defense.rename(columns={"defteam": "team"})
        def_rank_frames.append(defense[["season", "team", "def_rank"]])

    off_ranks = pd.concat(off_rank_frames, ignore_index=True)
    def_ranks = pd.concat(def_rank_frames, ignore_index=True)

    df = df.merge(
        off_ranks,
        left_on=["season", "posteam"],
        right_on=["season", "team"],
        how="left"
    ).drop(columns=["team"])

    df = df.merge(
        def_ranks,
        left_on=["season", "defteam"],
        right_on=["season", "team"],
        how="left"
    ).drop(columns=["team"])

    return df

#adding nonlinear terms and interaction terms for model tuning
def build_features(df):
    model_df = df.copy()

    # Nonlinear terms
    model_df["air_yards_sq"] = model_df["air_yards"] ** 2

    # Bunch of one-hot encodings for categorical variables

    #added bins for ydstogo to capture non-linear effects without overfitting. This is yards to go till 1st down. 
    model_df["ydstogo_bin"] = pd.cut(
        model_df["ydstogo"],
        bins=[0, 3, 7, 15, 30],
        labels=["short", "medium", "long", "very_long"]
    )
    yds_dummies = pd.get_dummies(model_df["ydstogo_bin"], prefix="yds", drop_first=True, dtype=int)
    model_df = pd.concat([model_df, yds_dummies], axis=1)
    
    loc_dummies = pd.get_dummies(model_df["pass_location"], prefix="loc", drop_first=True, dtype=int)
    down_dummies = pd.get_dummies(model_df["down"], prefix="down", drop_first=True, dtype=int)
    qtr_dummies = pd.get_dummies(model_df["qtr"], prefix="qtr", drop_first=True, dtype=int)

    model_df = pd.concat([model_df, loc_dummies, down_dummies, qtr_dummies], axis=1)

    # Interaction terms
    if "loc_middle" in model_df.columns:
        model_df["air_loc_middle"] = model_df["air_yards"] * model_df["loc_middle"]
    else:
        model_df["air_loc_middle"] = 0

    if "loc_right" in model_df.columns:
        model_df["air_loc_right"] = model_df["air_yards"] * model_df["loc_right"]
    else:
        model_df["air_loc_right"] = 0

    feature_cols = [
        "air_yards",
        "air_yards_sq",
        "yardline_100",
        "yds_medium",
        "yds_long",
        "yds_very_long",
        "shotgun",
        "off_rank",
        "def_rank",
        "loc_middle",
        "loc_right",
        "down_2",
        "down_3",
        "down_4",
        "qtr_2",
        "qtr_3",
        "qtr_4",
        "air_loc_middle",
        "air_loc_right",
    ]

    X = model_df[feature_cols].copy()
    y = model_df["complete_pass"].copy()

    return model_df, X, y, feature_cols



#### logistic regression ####
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=6242, stratify=y
    )

    model = LogisticRegression(max_iter=15000)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    print("\nLogistic Regression Performance")
    print("AUC    :", roc_auc_score(y_test, probs))
    print("LogLoss:", log_loss(y_test, probs))
    print("Brier  :", brier_score_loss(y_test, probs))

    return model, X_train, X_test, y_train, y_test, probs



def build_rank_lookups(df):
    off_rank_lookup = {}
    def_rank_lookup = {}

    off_df = df[["season", "posteam", "off_rank"]].drop_duplicates()
    def_df = df[["season", "defteam", "def_rank"]].drop_duplicates()

    for _, row in off_df.iterrows():
        off_rank_lookup[(int(row["season"]), row["posteam"])] = int(row["off_rank"])

    for _, row in def_df.iterrows():
        def_rank_lookup[(int(row["season"]), row["defteam"])] = int(row["def_rank"])

    return off_rank_lookup, def_rank_lookup

#joblib funcs
def save_artifacts(model, feature_cols, off_rank_lookup, def_rank_lookup):
    joblib.dump(model, MODEL_FILE)
    joblib.dump(feature_cols, FEATURES_FILE)
    joblib.dump(off_rank_lookup, OFF_RANK_FILE)
    joblib.dump(def_rank_lookup, DEF_RANK_FILE)


def load_artifacts():
    model = joblib.load(MODEL_FILE)
    feature_cols = joblib.load(FEATURES_FILE)
    off_rank_lookup = joblib.load(OFF_RANK_FILE)
    def_rank_lookup = joblib.load(DEF_RANK_FILE)
    return model, feature_cols, off_rank_lookup, def_rank_lookup

# This function takes in the model and the relevant features for a given pass attempt, 
# and outputs the predicted probability of completion. This is the main function 
# that would be used in a real-time application to get completion probabilities 
# for different pass scenarios.
def predict_pass_probability(air_yards, yardline_100, ydstogo, down, qtr,
                             shotgun,
                             off_rank, def_rank, pass_location):
    
    yds_short = int(ydstogo <= 3) #Baseline, so not included in row list. Just here for clarity
    yds_medium = int(ydstogo > 3 and ydstogo <= 7)
    yds_long = int(ydstogo > 7 and ydstogo <= 15)
    yds_very_long = int(ydstogo > 15)
    row = {
        "air_yards": air_yards,
        "air_yards_sq": air_yards ** 2,
        "yardline_100": yardline_100,
        "shotgun": int(shotgun),
        "off_rank": off_rank,
        "def_rank": def_rank,
        "loc_middle": int(pass_location == "middle"),
        "loc_right": int(pass_location == "right"),
        "down_2": int(down == 2),
        "down_3": int(down == 3),
        "down_4": int(down == 4),
        "qtr_2": int(qtr == 2),
        "qtr_3": int(qtr == 3),
        "qtr_4": int(qtr == 4),
        "yds_medium": yds_medium,
        "yds_long": yds_long,
        "yds_very_long": yds_very_long,
    }

    row["air_loc_middle"] = row["air_yards"] * row["loc_middle"]
    row["air_loc_right"] = row["air_yards"] * row["loc_right"]

    X_input = pd.DataFrame([row])
    X_input = X_input[feature_cols]

    prob = model.predict_proba(X_input)[0, 1]
    return prob




#helper function for flask integration. Call this func to get flask-friendly outputs.
def execute_pass_model(play):
    season = int(play.season)
    offense_team = play.offense_team
    defense_team = play.defense_team

    down = int(play.down)
    qtr = int(play.quarter)
    shotgun = int(play.shotgun)

    los = int(play.LOS)
    yardline_100 = 110 - los

    ydstogo = int(play.ydstogo)
    air_yards = int(play.pass_attempt_length)
    pass_location = play.pass_location

    off_rank = off_rank_lookup[(season, offense_team)]
    def_rank = def_rank_lookup[(season, defense_team)]

    prob = predict_pass_probability(
        air_yards=air_yards,
        yardline_100=yardline_100,
        ydstogo=ydstogo,
        down=down,
        qtr=qtr,
        shotgun=shotgun,
        off_rank=off_rank,
        def_rank=def_rank,
        pass_location=pass_location
    )

    return {
        "play_type": "pass",
        "probability": float(prob),
        "probability_percent": round(float(prob) * 100, 2)
    }


#training funcs
#df = load_data()
#df = clean_data(df)
#df = add_team_ranks(df)

#executing model funcs - looking up team ranks given season and team as flask inputs
#off_rank_lookup, def_rank_lookup = build_rank_lookups(df)

#model_df, X, y, feature_cols = build_features(df)


#model, X_train, X_test, y_train, y_test, probs = train_model(X, y)

#save_artifacts(model, feature_cols, off_rank_lookup, def_rank_lookup)

model, feature_cols, off_rank_lookup, def_rank_lookup = load_artifacts()

#this is a sample, fill in the variables as needed to produce an output probability for a given pass scenario. The variables are viewable in the function definition for predict_pass_probability.
#input_output = predict_pass_probability(3, 60, 2, 1, 2, 0, 8, 20, "middle")
#print("Sample output:", input_output)

#README#
#Run line 351 first to establish the model, cols, and rank lookups. This reads the .pkl files
#OPTIONAL TEST: uncomment and run line 354 and 355 to test the model w/ dummy inputs to validate functionality.
#execute_pass_model(dummy_play) where dummy_play is the flask input class object with the relevant play variables.
#output is a dict @ line 329-333

##input var for reference
# class DummyPlayClass:
#     def __init__(self):
#         self.season = 2018
#         self.offense_team = "KC"
#         self.defense_team = "BUF"
#         self.down = 3
#         self.quarter = 2
#         self.shotgun = 1
#         self.LOS = 35
#         self.ydstogo = 8
#         self.pass_attempt_length = 12
#         self.pass_location = "middle"

# dummy_play = DummyPlayClass()

# result = execute_pass_model(dummy_play)
# print("Helper output:", result)