import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('play_by_play_2020.csv')

df = df[(df["pass_attempt"] == 1) & (df["air_yards"].notna())]
if "play_deleted" in df: df = df[df["play_deleted"] == 0]
if "qb_spike" in df:df = df[df["qb_spike"] == 0]
if "qb_kneel" in df:df = df[df["qb_kneel"] == 0]

cols = [
    "air_yards",
    "complete_pass",
    "pass_location",
    "yardline_100",
    "ydstogo",
    "down",
    "qtr",
    "score_differential",
    "game_seconds_remaining",
    "shotgun"
]
df = df[cols].dropna()

#EDA plots
#hists
continuous_vars = [
    "air_yards",
    "yardline_100",
    "ydstogo",
    "score_differential",
    "game_seconds_remaining",
]
categorical_vars = ["pass_location", "down", "qtr"]

for i in continuous_vars:
    plt.figure(figsize=(6,4))
    df[i].hist(bins=40)
    plt.title(f"Histogram of {i}")
    plt.xlabel(i)
    plt.ylabel("Count")
    plt.show()


#completion percentage by continuous variables
for var in continuous_vars:
    temp = df[[var, "complete_pass"]].copy()
    temp["bin"] = pd.qcut(temp[var], q=10, duplicates="drop")
    grp = temp.groupby("bin", observed=False)["complete_pass"].mean()

    plt.figure(figsize=(7,4))
    grp.plot(marker="o")
    plt.title(f"Completion Rate by {var} (binned)")
    plt.xlabel(var)
    plt.ylabel("Completion Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#categorical variable counts
for var in categorical_vars:
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x=var)
    plt.title(f"Count Plot: {var}")
    plt.show()

#completion percentage by categorical variables
for var in categorical_vars:
    grp = df.groupby(var, observed=False)["complete_pass"].mean()

    plt.figure(figsize=(6,4))
    grp.plot(kind="bar")
    plt.title(f"Completion Rate by {var}")
    plt.ylabel("Completion Rate")
    plt.xticks(rotation=0)
    plt.show()


# completion % by air_yards AND field pos
df["air_bin"] = pd.cut(df["air_yards"], bins=range(-5, 51, 5))
pivot1 = df.pivot_table(
    values="complete_pass",
    index="air_bin",
    columns="pass_location",
    aggfunc="mean"
)

print("\nCompletion rate by air_yards & pass_location:")
print(pivot1)
plt.figure(figsize=(8,5))
sns.heatmap(pivot1, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Completion Rate: Air Yards & Pass Location")
plt.show()

#air yards by yard_line_100
df["yard_bin"] = pd.cut(df["yardline_100"], bins=range(0, 101, 10))
pivot2 = df.pivot_table(
    values="complete_pass",
    index="air_bin",
    columns="yard_bin",
    aggfunc="mean"
)
plt.figure(figsize=(10,5))
sns.heatmap(pivot2, cmap="YlGnBu")
plt.title("Completion Rate: Air Yards & Yardline")
plt.show()

#air yards by down
pivot3 = df.pivot_table(
    values="complete_pass",
    index="air_bin",
    columns="down",
    aggfunc="mean"
)
plt.figure(figsize=(7,5))
sns.heatmap(pivot3, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Completion Rate: Air Yards x Down")
plt.show()

###### VIF #######
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
vif_vars = [
    "air_yards",
    "yardline_100",
    "ydstogo",
    "score_differential",
    "game_seconds_remaining",
]

corr = df[vif_vars].corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Matrix heatmap")
plt.show()

X_vif = df[vif_vars].copy()
X_vif = sm.add_constant(X_vif)

vif_df = pd.DataFrame({
    "variable": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
print("\nVIF table:")
print(vif_df)

#### logistic regression ####
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve

X = df[[
    "air_yards",
    "yardline_100",
    "ydstogo",
    "down",
    "qtr",
    "score_differential",
    "game_seconds_remaining",
    "shotgun"
]].copy()
X = X.join(pd.get_dummies(df["pass_location"], drop_first=True))
y = df["complete_pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=6242, stratify=y
)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]

print("\Logistic Regression Performance")
print("AUC    :", roc_auc_score(y_test, probs))
print("LogLoss:", log_loss(y_test, probs))
print("Brier  :", brier_score_loss(y_test, probs))

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

frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)

plt.plot(mean_pred, frac_pos, marker="o")
plt.plot([0,1],[0,1],"--")
plt.title("Calibration")
plt.xlabel("Predicted")
plt.ylabel("Observed")
plt.show()

#linearity 
## retest w/ added non-linear squared terms
def logit(p):
    p = np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p))

for var in ["air_yards","yardline_100","ydstogo"]:
    temp = df[[var,"complete_pass"]].copy()
    temp["bin"] = pd.qcut(temp[var], q=10, duplicates="drop")
    grp = temp.groupby("bin")["complete_pass"].mean()
    
    plt.plot(range(len(grp)), logit(grp.values), marker="o")
    plt.title(f"logit linearity: {var}")
    plt.show()


#residuals
residuals = y_test - probs

plt.hist(residuals, bins=50)
plt.title("Residuals")
plt.show()