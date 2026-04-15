import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
#import matplotlib.pyplot as plt
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

#VIF check
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# import statsmodels.api as sm
# def print_vif(X):
#     X_vif = sm.add_constant(X)

#     vif_df = pd.DataFrame({
#         "variable": X_vif.columns,
#         "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
#     })

#     print("\nVIF")
#     print(vif_df.sort_values("VIF", ascending=False))

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

def plot_calibration(y_test, probs):
    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)

    plt.figure(figsize=(6, 4))
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], "--")
    plt.title("Calibration")
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.show()


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


df = load_data()
df = clean_data(df)
df = add_team_ranks(df)

model_df, X, y, feature_cols = build_features(df)


model, X_train, X_test, y_train, y_test, probs = train_model(X, y)

#this is a sample, fill in the variables as needed to produce an output probability for a given pass scenario. The variables are viewable in the function definition for predict_pass_probability.
input_output = predict_pass_probability(3, 60, 2, 1, 2, 0, 8, 20, "middle")

#testing and sample input/outputs

# --- High probability cases ---
# short_slant_first_down = predict_pass_probability(3, 60, 2, 1, 2, 0, 8, 20, "middle")
# print(f"short_slant_first_down: {short_slant_first_down:.2%}")

# short_out_second_down = predict_pass_probability(5, 50, 4, 2, 2, 1, 10, 18, "right")
# print(f"short_out_second_down: {short_out_second_down:.2%}")

# quick_middle_third_and_short = predict_pass_probability(4, 35, 3, 2, 3, 1, 12, 15, "middle")
# print(f"quick_middle_third_and_short: {quick_middle_third_and_short:.2%}")


# # --- Average probability cases ---
# intermediate_second_and_long = predict_pass_probability(10, 45, 8, 2, 2, 1, 14, 14, "middle")
# print(f"intermediate_second_and_long: {intermediate_second_and_long:.2%}")

# third_and_medium_normal_pass = predict_pass_probability(12, 40, 10, 3, 2, 1, 12, 12, "right")
# print(f"third_and_medium_normal_pass: {third_and_medium_normal_pass:.2%}")

# mid_range_vs_strong_defense = predict_pass_probability(8, 30, 6, 2, 3, 0, 16, 10, "middle")
# print(f"mid_range_vs_strong_defense: {mid_range_vs_strong_defense:.2%}")

# early_down_balanced_teams = predict_pass_probability(7, 25, 5, 1, 1, 0, 18, 18, "right")
# print(f"early_down_balanced_teams: {early_down_balanced_teams:.2%}")


# # --- Low probability cases ---
# deep_third_and_long = predict_pass_probability(18, 50, 12, 3, 3, 1, 15, 10, "right")
# print(f"deep_third_and_long: {deep_third_and_long:.2%}")

# fourth_down_deep_attempt = predict_pass_probability(20, 35, 15, 4, 4, 1, 18, 8, "right")
# print(f"fourth_down_deep_attempt: {fourth_down_deep_attempt:.2%}")

# red_zone_tight_window_strong_defense = predict_pass_probability(15, 20, 10, 3, 4, 1, 20, 5, "right")
# print(f"red_zone_tight_window_strong_defense: {red_zone_tight_window_strong_defense:.2%}")




#visuals and testing stuff
#print_vif(X)

#plot_calibration(y_test, probs)

# def logit(p):
#     p = np.clip(p, 1e-6, 1 - 1e-6)
#     return np.log(p / (1 - p))


# vars_to_check = ["air_yards", "yardline_100", "ydstogo"]

# for var in vars_to_check:
#     temp = model_df[[var, "complete_pass"]].copy()
#     temp["bin"] = pd.qcut(temp[var], q=10, duplicates="drop")
#     grp = temp.groupby("bin", observed=False)["complete_pass"].mean()

#     plt.figure(figsize=(6, 4))
#     plt.plot(range(len(grp)), logit(grp.values), marker="o")
#     plt.title(f"Logit Linearity: {var}")
#     plt.xlabel("Bin")
#     plt.ylabel("Logit(Completion Rate)")
#     plt.show()

# residuals = y_test - probs

# plt.figure(figsize=(6, 4))
# plt.hist(residuals, bins=50)
# plt.title("Residuals")
# plt.xlabel("Residual")
# plt.ylabel("Count")
# plt.show()


# import statsmodels.api as sm

# X_sm = sm.add_constant(X)
# result = sm.Logit(y, X_sm).fit()

# print(result.summary())




#to-do's from 4/1/2026 meeting
#model tuning:
### add squared air yards and yardline_100 as non-linear interaction terms.
### add left and right pass location interaction terms.
### 

#variable selection/trimming
#data cleaning 2nd pass:
### remove 5th & 6th quarter
### try model tuning with one-hot encoding of down, pass location, and quarter
### remove incorrect null rows
### remove non-pass or incomplete plays (check penalties)
#output format: left, middle, right probability values for a given air_yard input

#model assumption and calibration testings