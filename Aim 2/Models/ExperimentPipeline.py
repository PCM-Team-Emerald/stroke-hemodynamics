# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 20:29:06 2021

@author: mainswo3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle
import sys

sys.path.insert(
    0,
    "S:\Dehydration_stroke\Team Emerald\Working GitHub Directories\Michael\stroke-hemodynamics\Aim 2\Models",
)
import Classifiers


def print_metrics(y_test, predictions):
    """
    Print our confusion matrix and evaluation stats given y_test and model 
    predictions.
    """
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))


def get_auc_pr(y_test, raw_predictions):
    """
    Given predictions made by model and the actual labels, return metrics
    needed to calculate roc and precision recall curves.
    """
    fpr, tpr, roc_thresholds = roc_curve(y_test_p, raw_predictions[:, 1])
    auc_values = roc_auc_score(y_test, raw_predictions[:, 1])
    precisions, recalls, _ = precision_recall_curve(y_test, raw_predictions[:, 1])
    pr_auc_values = auc(recalls, precisions)
    return auc_values, pr_auc_values, fpr, tpr, roc_thresholds, recalls, precisions


def optimal_points(fpr, tpr, raw_predictions, roc_thresholds):
    """
    Using precision recall metrics, determine the optimal operating point 
    predictions and threshold.
    """
    min_dist = np.inf
    threshold = 0
    for i in range(len(fpr)):
        dist = np.sqrt((1 - tpr[i]) ** 2 + fpr[i] ** 2)
        if dist < min_dist:
            min_dist = dist
            threshold = roc_thresholds[i]
    optimal_preds = np.where(raw_predictions[:, 1] < threshold, 0, 1)
    return optimal_preds, threshold


def getCat(x):
    """
    Categorization for feature ranking functions
    """
    if "orientation" in x:
        return "orientation"
    elif "conciousness" in x:
        return "conciousness"
    elif "ampac" in x:
        return "ampac"
    elif "pulse" in x:
        return "pulse"
    elif "temp" in x:
        return "temp"
    elif "glasgow" in x:
        return "glasgow"
    elif "iv" in x:
        return "iv"
    else:
        return "other"


def featureRankGLM(X_train_p, X_train_df, y_train):
    """
    Perform feature ranking using a GLM. Figure will be generated depicting the
    top 20 scoring features. Returns variable to create new feature space and 
    variables to edit figure.
    
    # source: kaggle.com user: dkim1992
    """
    clf = LogisticRegression(penalty="l1", C=0.1, solver="liblinear", max_iter=200)
    clf.fit(X_train_p, y_train)

    num_features = X_train_df.shape[1]
    zero_feat = []
    nonzero_feat = []
    for i in range(num_features):
        coef = clf.coef_[0, i]
        if coef == 0:
            zero_feat.append(X_train_df.columns[i])
        else:
            nonzero_feat.append((coef, X_train_df.columns[i]))

    nznew = sorted(nonzero_feat, reverse=True)

    target = nznew[:20]
    glm_coefs = pd.DataFrame(data=target, columns=["val", "feat"])
    glm_coefs = glm_coefs.iloc[glm_coefs["val"].abs().argsort()].iloc[::-1]
    glm_coefs["cat"] = glm_coefs["feat"].apply(lambda x: getCat(x))

    colors = {
        "orientation": "tab:blue",
        "conciousness": "tab:orange",
        "ampac": "tab:green",
        "pulse": "tab:olive",
        "temp": "tab:purple",
        "glasgow": "tab:red",
        "iv": "tab:brown",
        "other": "pink",
    }

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.bar(
        data=glm_coefs,
        x=range(20),
        height="val",
        color=[colors[x] for x in glm_coefs.head(20)["cat"].tolist()],
    )
    lines = [
        "orientation",
        "conciousness",
        "ampac",
        "pulse",
        "temp",
        "glasgow",
        "iv",
        "other",
    ]
    custom_lines = [Line2D([0], [0], color=colors[x], lw=4) for x in lines]

    ax.legend(custom_lines, lines)

    plt.xlabel("Feature", fontsize=18)
    plt.ylabel("GLM Coefficient", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    return nznew, fig, ax


def featureRankingRF(X_train_p, X_train_df, y_train):
    """
    Perform feature ranking using a RF. Figure will be generated depicting the
    top 20 scoring features. Returns variable to create new feature space and 
    variables to edit figure.
    """
    feat_labels = X_train_df.columns

    clf = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features=2,
    )
    clf.fit(X_train_p, y_train_p)

    rf_features = []
    for feature in zip(feat_labels, clf.feature_importances_):
        rf_features.append(feature)

    new = sorted(rf_features, key=lambda x: x[1], reverse=True)

    target = new[:20]

    rf_coefs = pd.DataFrame(data=target, columns=["feat", "val"])
    rf_coefs = rf_coefs.iloc[rf_coefs["val"].abs().argsort()].iloc[::-1]
    rf_coefs["cat"] = rf_coefs["feat"].apply(lambda x: getCat(x))

    colors = {
        "orientation": "tab:blue",
        "conciousness": "tab:orange",
        "ampac": "tab:green",
        "pulse": "tab:olive",
        "temp": "tab:purple",
        "glasgow": "tab:red",
        "iv": "tab:brown",
        "other": "pink",
    }

    fig = plt.figure()
    ax = plt.subplot(111)
    plt.bar(
        data=rf_coefs,
        x=range(20),
        height="val",
        color=[colors[x] for x in rf_coefs.head(20)["cat"].tolist()],
    )

    lines = [
        "orientation",
        "conciousness",
        "ampac",
        "pulse",
        "temp",
        "glasgow",
        "iv",
        "other",
    ]

    custom_lines = [Line2D([0], [0], color=colors[x], lw=4) for x in lines]

    plt.legend(custom_lines, lines)
    plt.xlabel("Feature", fontsize=18)
    plt.ylabel("RF Score", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Retrain model with only top features
    sfm = SelectFromModel(clf, threshold=0.0003)
    sfm.fit(X_train, y_train)

    return sfm, fig, ax


##############################
######## Load Dataset ########

# Select data to run
#path = "C:\\Users\\mainswo3\\Downloads\\complete_24h_norehab_new.csv"

# path = 'C:\\Users\\mainswo3\\Downloads\\complete_24h_norehab_new.csv'
# path = 'C:\\Users\\mainswo3\\Downloads\\complete_48h_norehab_new.csv'
# path = 'C:\\Users\\mainswo3\\Downloads\\complete_72h_norehab_new.csv'

# Use below path to run with GCS data
path = 'C:\\Users\\mainswo3\\Downloads\\complete_24h_min_gcs_new.csv'
# path = 'C:\\Users\\mainswo3\\Downloads\\complete_24h_72h_min_gcs_new.csv'

df = pd.read_csv(path)
df = df.drop('Unnamed: 0', axis=1)

####
# Use if there is null values!!
# df = df.dropna()
####

# Create X and y varables, remove label from X data
# y = df["LOS"]
# X = df.drop(["LOS", "mrn_csn_pair"], axis=1)
y = df["bin_min_gcs"]
X = df.drop(["bin_min_gcs", "mrn_csn_pair"], axis=1)
print("Shape of X data: ", X.shape)
print("Shape of X data: ", y.shape)


# Run a train test split on the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_train_p = preprocessing.scale(X_train)
X_test_p = preprocessing.scale(X_test)
y_train_p = np.squeeze(y_train.to_numpy())
y_test_p = np.squeeze(y_test.to_numpy())


######## GLM Hyperparameter Tuning ########
hyperparameter_glm = dict()
hyperparameter_glm["C"] = [0.001, 0.01, 0.1, 1, 10, 100]

logistic = LogisticRegression(solver="lbfgs", max_iter=200, penalty="l2")
randomizedsearch = RandomizedSearchCV(logistic, hyperparameter_glm, cv=5, n_jobs=-1)
best_model_random = randomizedsearch.fit(X_train, y_train)
best_params_glm = best_model_random.best_params_


######## RF Hyperparameter Tuning ########
hyperparameter_rf = dict()
hyperparameter_rf["min_samples_split"] = [1.0, 2, 3, 4]
hyperparameter_rf["min_samples_leaf"] = [0.1, 0.3, 0.4, 1]
hyperparameter_rf["max_features"] = ["auto", "sqrt", "log2"]

rf = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=None)
randomizedsearch = RandomizedSearchCV(rf, hyperparameter_rf, cv=5, n_jobs=-1)
best_model = randomizedsearch.fit(X_train, y_train)
print("\n\n\nRF BEST PARAMS:\n\n\n")
best_params_rf = best_model.best_params_
print(best_params_rf)


######## XGB Hyperparameter Tuning ########
hyperparameter_xg = dict()
hyperparameter_xg["booster"] = ["gbtree", "gblinear", "dart"]
hyperparameter_xg["eta"] = [0.1, 0.2, 0.3, 1]
hyperparameter_xg["max_depth"] = [4, 6, 8]

xg = XGBClassifier(n_estimators=100, criterion="gini", max_depth=None)
randomizedsearch = RandomizedSearchCV(xg, hyperparameter_xg, cv=5, n_jobs=-1)
best_model_xg = randomizedsearch.fit(X_train, y_train)
print("\n\n\nXGB BEST PARAMS:\n\n\n")
best_params_xg = best_model_xg.best_params_
print(best_params_xg)


########################################
######## Print Original Results ########
rf_anova = Classifiers.RandomForestModel(
    params={
        **best_params_rf,
        "n_estimators": 100,
        "criterion": "gini",
        "max_depth": None,
        "verbose": 0,
    }
)
rf_anova.fit(X_train_p, y_train_p)
rf_anova_raw_preds, rf_anova_preds, rf_anova_score = rf_anova.predict(
    X_test_p, y_test_p
)


auc_rf, pr_auc, fpr, tpr, roc_thresholds, recalls, precisions = get_auc_pr(
    y_test_p, rf_anova_raw_preds
)

glm_anova = Classifiers.LogisticRegressionModel(
    params={
        **best_params_glm,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 200,
        #            'C' : 0.001,
        "verbose": 0,
    }
)
glm_anova.fit(X_train_p, y_train_p)
glm_anova_raw_preds, glm_anova_preds, glm_anova_score = glm_anova.predict(
    X_test_p, y_test_p
)


auc_glm, pr_auc_glm, fpr_glm, tpr_glm, roc_thresholds_glm, recalls_glm, precisions_glm = get_auc_pr(
    y_test_p, glm_anova_raw_preds
)


xgb = Classifiers.XGBoostModel(params={**best_params_xg})
xgb.fit(X_train_p, y_train_p)
xgb_raw_preds, xgb_preds, xgb_score = xgb.predict(X_test_p, y_test_p)


auc_xgb, pr_auc_xgb, fpr_xgb, tpr_xgb, roc_thresholds_xgb, recalls_xgb, precisions_xgb = get_auc_pr(
    y_test_p, xgb_raw_preds
)


recall_no_skill = y_train_p.sum() / len(y_train_p)


optimal_preds, threshold = optimal_points(fpr, tpr, rf_anova_raw_preds, roc_thresholds)
optimal_preds_glm, threshold_glm = optimal_points(
    fpr_glm, tpr_glm, glm_anova_raw_preds, roc_thresholds_glm
)
optimal_preds_xgb, threshold_xgb = optimal_points(
    fpr_xgb, tpr_xgb, xgb_raw_preds, roc_thresholds_xgb
)


fig = plt.figure(figsize=(12, 6))
ax = plt.subplot(1, 2, 1)
ax.plot(fpr, tpr, lw=2.5, c="green", alpha=0.7)
ax.plot(fpr_glm, tpr_glm, lw=2.5, c="red", alpha=0.7)
ax.plot(fpr_xgb, tpr_xgb, lw=2.5, c="blue", alpha=0.7)
ax.plot([0, 1], [0, 1], ls="--", color="black")
plt.title("ROC Curve", fontsize=20)
plt.xlabel("FPR", fontsize=18)
plt.ylabel("TPR", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(
    [
        "RF AUC = {}".format(auc_rf.round(3)),
        "GLM AUC = {}".format(auc_glm.round(3)),
        "XGB AUC = {}".format(auc_xgb.round(3)),
        "No-Skill",
    ],
    loc="lower right",
    fontsize=16,
)

ax2 = plt.subplot(1, 2, 2)
ax2.plot(recalls, precisions, lw=2.5, c="green", alpha=0.7)
ax2.plot(recalls_glm, precisions_glm, lw=2.5, c="red", alpha=0.7)
ax2.plot(recalls_xgb, precisions_xgb, lw=2.5, c="blue", alpha=0.7)
ax2.plot([0, 1], [recall_no_skill, recall_no_skill], ls="--", color="black")
plt.title("Precision-Recall Curve", fontsize=18)
plt.xlabel("Recall", fontsize=18)
plt.ylabel("Precision", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
plt.legend(
    [
        "RF AUC = {}".format(pr_auc.round(3)),
        "GLM AUC = {}".format(pr_auc_glm.round(3)),
        "XGB AUC = {}".format(pr_auc_xgb.round(3)),
        "No-Skill",
    ],
    loc="lower left",
    fontsize=16,
)
plt.tight_layout()
plt.show()
# fig.savefig('S:/Dehydration_stroke/Team Emerald/scripts/Michael Working Scripts/Michael Figures/AllFeaturesHyperparameter.png')


# Perform GLM feature ranking
nznew, fig, ax = featureRankGLM(X_train_p, X_train, y_train_p)

plt.title("GLM Top 20 Features (24 Hours)", fontsize=20)
plt.show()

features_used = [i[1] for i in nznew]
X_glm_train = X_train[features_used]
X_glm_test = X_test[features_used]

# Scale and normalize
X_train_new = preprocessing.scale(X_glm_train)
X_test_new = preprocessing.scale(X_glm_test)


# Re-tune hyperparameters with new feature space
######## GLM Hyperparameter Tuning ########
hyperparameter_glm = dict()
hyperparameter_glm["C"] = [0.001, 0.01, 0.1, 1, 10, 100]

logistic = LogisticRegression(solver="lbfgs", max_iter=200, penalty="l2")
randomizedsearch = RandomizedSearchCV(logistic, hyperparameter_glm, cv=5, n_jobs=-1)
best_model_random = randomizedsearch.fit(X_train_new, y_train)
best_params_glm_updated = best_model_random.best_params_
print("BP_GLM: ", best_params_glm_updated)


# Use new features to train GLM model
glm = Classifiers.LogisticRegressionModel(
    params={
        **best_params_glm_updated,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 200,
        "verbose": 0,
    }
)
glm.fit(X_train_new, y_train_p)

filename = 'S:/Dehydration_stroke/Team Emerald/Working GitHub Directories/'\
           'Michael/stroke-hemodynamics/Aim 2/Models/FullModelResults/gcs_24hr_model_glm.sav'
pickle.dump(glm, open(filename, 'wb'))

glm_raw_preds, glm_preds, glm_score = glm.predict(X_test_new, y_test_p)
auc_glm, pr_auc_glm, fpr_glm, tpr_glm, roc_thresholds_glm, recalls_glm, precisions_glm = get_auc_pr(
    y_test_p, glm_raw_preds
)

recall_no_skill = y_train_p.sum() / len(y_train_p)
optimal_preds_glm, threshold_glm = optimal_points(
    fpr_glm, tpr_glm, glm_raw_preds, roc_thresholds_glm
)

print_metrics(y_test_p, optimal_preds_glm)


# Perform RF feature ranking
sfm, fig, ax = featureRankingRF(X_train_p, X_train, y_train_p)
plt.title("RF Top 20 Features (24 Hours)", fontsize=20)
plt.show()


X_train_new = sfm.transform(X_train)
X_test_new = sfm.transform(X_test)


# Retune hyperparameters with new feature space
######## RF Hyperparameter Tuning ########
hyperparameter_rf = dict()
hyperparameter_rf["min_samples_split"] = [1.0, 2, 3, 4]
hyperparameter_rf["min_samples_leaf"] = [0.1, 0.3, 0.4, 1]
hyperparameter_rf["max_features"] = ["auto", "sqrt", "log2"]

rf = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=None)
randomizedsearch = RandomizedSearchCV(rf, hyperparameter_rf, cv=5, n_jobs=-1)
best_model = randomizedsearch.fit(X_train_new, y_train)
print("\n\n\nRF BEST PARAMS:\n\n\n")
best_params_rf_updated = best_model.best_params_
print(best_params_rf_updated)


######## XGB Hyperparameter Tuning ########
hyperparameter_xg = dict()
hyperparameter_xg["booster"] = ["gbtree", "gblinear", "dart"]
hyperparameter_xg["eta"] = [0.1, 0.2, 0.3, 1]
hyperparameter_xg["max_depth"] = [4, 6, 8]

xg = XGBClassifier(n_estimators=100, criterion="gini", max_depth=None)
randomizedsearch = RandomizedSearchCV(xg, hyperparameter_xg, cv=5, n_jobs=-1)
best_model_xg = randomizedsearch.fit(X_train_new, y_train)
print("\n\n\nXGB BEST PARAMS:\n\n\n")
best_params_xg_updated = best_model_xg.best_params_
print(best_params_xg_updated)


# Scale and normalize raw data
X_train_p = preprocessing.scale(X_train_new)
X_test_p = preprocessing.scale(X_test_new)
y_train_p = np.squeeze(y_train.to_numpy())
y_test_p = np.squeeze(y_test.to_numpy())


# Use new features to train RF and XGB models
rf = Classifiers.RandomForestModel(
    params={
        **best_params_rf_updated,
        "n_estimators": 100,
        "criterion": "gini",
        "max_depth": None,
        "verbose": 0,
    }
)
rf.fit(X_train_p, y_train_p)

filename = 'S:/Dehydration_stroke/Team Emerald/Working GitHub Directories/'\
           'Michael/stroke-hemodynamics/Aim 2/Models/FullModelResults/gcs_24hr_model_rf.sav'
pickle.dump(rf, open(filename, 'wb'))

rf_raw_preds, rf_preds, rf_score = rf.predict(X_test_p, y_test_p)

auc_rf, pr_auc_rf, fpr, tpr, roc_thresholds, recalls, precisions = get_auc_pr(
    y_test_p, rf_raw_preds
)


xgb = Classifiers.XGBoostModel(params={**best_params_xg_updated})
xgb.fit(X_train_p, y_train_p)

filename = 'S:/Dehydration_stroke/Team Emerald/Working GitHub Directories/'\
           'Michael/stroke-hemodynamics/Aim 2/Models/FullModelResults/gcs_24hr_model_xgb.sav'
pickle.dump(xgb, open(filename, 'wb'))

xgb_raw_preds, xgb_preds, xgb_score = xgb.predict(X_test_p, y_test_p)

auc_xgb, pr_auc_xgb, fpr_xgb, tpr_xgb, roc_thresholds_xgb, recalls_xgb, precisions_xgb = get_auc_pr(
    y_test_p, xgb_raw_preds
)

recall_no_skill_24 = y_train_p.sum() / len(y_train_p)

optimal_preds, threshold = optimal_points(fpr, tpr, rf_raw_preds, roc_thresholds)
optimal_preds_xgb, threshold_xgb = optimal_points(
    fpr_xgb, tpr_xgb, xgb_raw_preds, roc_thresholds_xgb
)

print_metrics(y_test_p, optimal_preds)
print_metrics(y_test_p, optimal_preds_xgb)


# Plot XGBoost and RF; lines generated from RF feature ranking
fig = plt.figure(figsize=(12, 6))
ax = plt.subplot(1, 2, 1)
ax.plot(fpr, tpr, lw=2.5, c="green", alpha=0.7)
ax.plot(fpr_xgb, tpr_xgb, lw=2.5, c="blue", alpha=0.7)
ax.plot([0, 1], [0, 1], ls="--", color="black")
plt.title("ROC Curve", fontsize=20)
plt.xlabel("FPR", fontsize=18)
plt.ylabel("TPR", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(
    [
        "RF AUC = {}".format(auc_rf.round(3)),
        "XGB AUC = {}".format(auc_xgb.round(3)),
        "No-Skill",
    ],
    loc="lower right",
    fontsize=16,
)

ax2 = plt.subplot(1, 2, 2)
ax2.plot(recalls, precisions, lw=2.5, c="green", alpha=0.7)
ax2.plot(recalls_xgb, precisions_xgb, lw=2.5, c="blue", alpha=0.7)
ax2.plot([0, 1], [recall_no_skill, recall_no_skill], ls="--", color="black")
plt.title("Precision-Recall Curve", fontsize=18)
plt.xlabel("Recall", fontsize=18)
plt.ylabel("Precision", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
plt.legend(
    [
        "RF AUC = {}".format(pr_auc_rf.round(3)),
        "XGB AUC = {}".format(pr_auc_xgb.round(3)),
        "No-Skill",
    ],
    loc="lower left",
    fontsize=16,
)
plt.tight_layout()
plt.show()


# Plot only XGBoost which generally performs best

fig = plt.figure(figsize=(12, 6))
ax = plt.subplot(1, 2, 1)
ax.plot(fpr_xgb, tpr_xgb, lw=2.5, c="blue", alpha=0.7)
ax.plot([0, 1], [0, 1], ls="--", color="black")
plt.title("ROC Curve", fontsize=20)
plt.xlabel("FPR", fontsize=18)
plt.ylabel("TPR", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(
    ["XGB AUC = {}".format(auc_xgb.round(3)), "No-Skill"],
    loc="lower right",
    fontsize=16,
)

ax2 = plt.subplot(1, 2, 2)
ax2.plot(recalls_xgb, precisions_xgb, lw=2.5, c="blue", alpha=0.7)
ax2.plot([0, 1], [recall_no_skill, recall_no_skill], ls="--", color="black")
plt.title("Precision-Recall Curve", fontsize=18)
plt.xlabel("Recall", fontsize=18)
plt.ylabel("Precision", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
plt.legend(
    ["XGB AUC = {}".format(pr_auc_xgb.round(3)), "No-Skill"],
    loc="lower left",
    fontsize=16,
)
plt.tight_layout()
plt.show()


# Combined Plot; plot all three final lines together

fig = plt.figure(figsize=(10, 5))
ax = plt.subplot(1, 2, 1)
ax.plot(fpr, tpr, lw=2.5, c="green", alpha=0.7)
ax.plot(fpr_xgb, tpr_xgb, lw=2.5, c="blue", alpha=0.7)
ax.plot(fpr_glm, tpr_glm, lw=2.5, c="red", alpha=0.7)
ax.plot([0, 1], [0, 1], ls="--", color="black")
plt.title("ROC Curve", fontsize=20)
plt.xlabel("FPR", fontsize=18)
plt.ylabel("TPR", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(
    [
        "RF AUC = {}".format(auc_rf.round(3)),
        "XGB AUC = {}".format(auc_xgb.round(3)),
        "GLM AUC = {}".format(auc_glm.round(3)),
        "No-Skill",
    ],
    loc="lower right",
    fontsize=14,
)

ax2 = plt.subplot(1, 2, 2)
ax2.plot(recalls, precisions, lw=2.5, c="green", alpha=0.7)
ax2.plot(recalls_xgb, precisions_xgb, lw=2.5, c="blue", alpha=0.7)
ax2.plot(recalls_glm, precisions_glm, lw=2.5, c="red", alpha=0.7)
ax2.plot([0, 1], [recall_no_skill, recall_no_skill], ls="--", color="black")
plt.title("Precision-Recall Curve", fontsize=22)
plt.xlabel("Recall", fontsize=20)
plt.ylabel("Precision", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
plt.legend(
    [
        "RF AUC = {}".format(pr_auc_rf.round(3)),
        "XGB AUC = {}".format(pr_auc_xgb.round(3)),
        "GLM AUC = {}".format(pr_auc_glm.round(3)),
        "No-Skill",
    ],
    loc="lower left",
    fontsize=14,
)
plt.tight_layout()
plt.show()

# fig.savefig('S:/Dehydration_stroke/Team Emerald/Working GitHub Directories/'\
#             'Michael/stroke-hemodynamics/Aim 2/Models/FullModelResults/PaperFigure3.png',
#            dpi=800)
    