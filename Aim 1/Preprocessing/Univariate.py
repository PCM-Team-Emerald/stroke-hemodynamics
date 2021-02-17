import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif


# Import csv containing finalized preprocessed data.
# Note: This may change to directly importing from the database.
path = 'S:\Dehydration_stroke\Team Emerald\Working Data\Preprocessed\Working\Complete_db.csv'
df = pd.read_csv(path)
df = df.drop('Unnamed: 0', axis=1)

# Create X and y varables, remove label from X data
y = df['LOS']
X = df.drop('LOS', axis=1)

# remove features we wouldnt have at admission
X = X.drop(['discharge_floor', 'discharge_ICU', 'discharge_stroke_unit'], axis=1)

# Drop GCS tsfresh features??
X = X.drop(list(X.filter(regex='glasgow')), axis=1)

# LDA Features
feat_drop = ['airway_airway','art_line', 'catheter', 'chemo', 'cvc_line', 'device', 'drain', 'epidural_line',
            'intraosseous_line', 'line', 'picc_line', 'piv_line', 'tube', 'urine_ostomy', 'wound']
X = X.drop(feat_drop, axis=1)

# Include the below code to only run models on time series features
# static_features = df.columns[-49:]
# Remove all non-time series features
# X = X.drop(static_features, axis=1)

print(X.shape)
print(y.shape)

# Run a train test split on the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Scale and normalize raw data
X_train_p = preprocessing.scale(X_train)
X_test_p = preprocessing.scale(X_test)

y_train_p = np.squeeze(y_train.to_numpy())
y_test_p = np.squeeze(y_test.to_numpy())



# VARIANCE THRESHOLD

vt = VarianceThreshold(threshold=(1000))
X_train_vt = vt.fit_transform(X_train)

m_ind = vt.get_support(indices=True)
vt_features_included = X_train.columns[m_ind]

X_train_vt = X_train[vt_features_included]
X_test_vt = X_test[vt_features_included]

print(vt_features_included)



# MUTUAL INFORMATION

select_k_best = 100

mi_results = SelectKBest(mutual_info_classif, k=select_k_best).fit(X_train, y_train)
mi_features = X_train.columns[mi_results.get_support()]

print('Retained Features: ',mi_features)

X_train_mi = X_train[mi_features]
X_test_mi = X_test[mi_features]


all_mi = mutual_info_classif(X_train, y_train, random_state = 0)
X_data = pd.concat([pd.DataFrame(X_train_p), pd.DataFrame(y_train_p)], axis=1)
header = X_data.columns.tolist()
features = header[0:len(header)-1]
names_scores = {'Names':X_train.columns[features], 'Scores':all_mi}
mi_output = pd.DataFrame(names_scores)
mi_output = mi_output.sort_values(by='Scores')
mi_output = mi_output.reindex(index=mi_output.index[::-1])
print('Mutual Information Scores for all Features: ',mi_output)

fig = plt.figure(figsize = (7,5))
ax = plt.subplot(111)
sns.distplot(mi_output['Scores'], kde=False, color='green')
plt.title('Mutual Information Scores', fontsize=20)
plt.xlabel('Count', fontsize=18)
plt.ylabel('Mutual Information',fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()



# CORRELATION FILTER

corr_features = set()

df_corr = df.drop(list(df.filter(regex='glasgow')), axis=1)
df_corr = df_corr.drop(feat_drop, axis=1)
# df_corr = df_corr.drop(static_features, axis=1)

correlations = df_corr.corr()

vs_los = correlations['LOS'].sort_values().dropna()
vs_los = vs_los.drop('LOS', axis=0)
# print(vs_los)

# Remove highly correlated features
for i in range(len(correlations.columns)):
    for j in range(i):
        if abs(correlations.iloc[i,j]) > 0.8:
            colname = correlations.columns[i]
            corr_features.add(colname)
            
X_train_corr = X_train.drop(labels = corr_features, axis=1)
X_test_corr = X_test.drop(labels = corr_features, axis=1)
print(X_train_corr)

fig = plt.figure(figsize = (6,5))
ax = plt.subplot(111)
# ax.xaxis.set_visible(False)
ax.set_xticks([])
ax.plot(vs_los, color='green', lw=4)
plt.title('Correlation of Features vs LOS',fontsize=20)
plt.xlabel('All Raw Data Features',fontsize=18)
plt.ylabel('Correlation to LOS',fontsize=18)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()



# CHI SQUARED

positive_X_train = X_train.loc[:, df.ge(0).all()]

select_k_features = 100
chi_results = SelectKBest(chi2, k=select_k_features).fit(positive_X_train, y_train)

chi_features = positive_X_train.columns[chi_results.get_support()]

X_train_chi = X_train[chi_features]
X_test_chi = X_test[chi_features]

print(chi_features)



# ANOVA

select_k_best = 5

anova_results = SelectKBest(f_classif, k=select_k_best).fit(X_train, y_train)

anova_features = X_train.columns[anova_results.get_support()]
print(anova_features)

X_train_anova = X_train[anova_features]
X_test_anova = X_test[anova_features]


all_anova = f_classif(X_train, y_train)

X_data = pd.concat([pd.DataFrame(X_train_p), pd.DataFrame(y_train_p)], axis=1)
header = X_data.columns.tolist()
features = header[0:len(header)-1]
names_scores = {'Names':X_train.columns[features], 'Scores':all_anova[0]}
# print(len(all_anova))
# print(X_train.columns[features].shape)
# print(names_scores)
anova_f = pd.DataFrame(names_scores)
anova_f = anova_f.sort_values(by='Scores')
anova_f = anova_f.reindex(index=mi_output.index[::-1])
print('ANOVA Scores for all Features: ',mi_output)



# Scale and normalize all data from post feature selection

X_train_vt_p = preprocessing.scale(X_train_vt)
X_test_vt_p = preprocessing.scale(X_test_vt)

X_train_mi_p = preprocessing.scale(X_train_mi)
X_test_mi_p = preprocessing.scale(X_test_mi)

X_train_corr_p = preprocessing.scale(X_train_corr)
X_test_corr_p = preprocessing.scale(X_test_corr)

X_train_chi_p = preprocessing.scale(X_train_chi)
X_test_chi_p = preprocessing.scale(X_test_chi)

X_train_anova_p = preprocessing.scale(X_train_anova)
X_test_anova_p = preprocessing.scale(X_test_anova)



# Import functions for running random forest and GLM
# Note: In this code, we created figures manually. To quickly generate ROC and 
# Precision-Recall Curves, use the function Models.metrics

import sys
sys.path.append("S:\Dehydration_stroke\Team Emerald\Working GitHub Directories\Michael\stroke-hemodynamics\Aim 2")
import Models

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

glm_anova = Models.LogisticRegressionModel()
glm_anova.fit(X_train_anova_p, y_train_p)

glm_anova_raw_preds, glm_anova_preds, glm_anova_score = glm_anova.predict(X_test_anova_p, y_test_p)

rf_anova = Models.RandomForestModel()
rf_anova.fit(X_train_anova_p, y_train_p)

rf_anova_raw_preds, rf_anova_preds, rf_anova_score = rf_anova.predict(X_test_anova_p, y_test_p)

fpr, tpr, roc_thresholds = roc_curve(y_test_p, rf_anova_raw_preds[:,1])
auc = roc_auc_score(y_test_p, rf_anova_raw_preds[:,1])
precisions, recalls, _ = precision_recall_curve(y_test_p, rf_anova_raw_preds[:,1])
print(confusion_matrix(y_test_p, rf_anova_preds))
print(classification_report(y_test_p, rf_anova_preds))


fpr_glm, tpr_glm, roc_thresholds_glm = roc_curve(y_test_p, glm_anova_raw_preds[:,1])
auc_glm = roc_auc_score(y_test_p, glm_anova_raw_preds[:,1])
precisions_glm, recalls_glm, _ = precision_recall_curve(y_test_p, glm_anova_raw_preds[:,1])
print(confusion_matrix(y_test_p, glm_anova_preds))
print(classification_report(y_test_p, glm_anova_preds))


recall_no_skill = y_test_p.sum()/len(y_test_p)


min_dist = np.inf
threshold = 0
for i in range(len(fpr)):
    dist = np.sqrt((1-tpr[i])**2 + fpr[i]**2)
    if dist < min_dist:
        min_dist = dist
        threshold = roc_thresholds[i]
optimal_preds = np.where(rf_anova_raw_preds[:,1] < threshold, 0, 1)
print(threshold.round(3))
print(confusion_matrix(y_test_p, optimal_preds))
print(classification_report(y_test_p, optimal_preds))

min_dist_glm = np.inf
threshold_glm = 0
for i in range(len(fpr_glm)):
    dist_glm = np.sqrt((1-tpr_glm[i])**2 + fpr_glm[i]**2)
    if dist_glm < min_dist_glm:
        min_dist_glm = dist_glm
        threshold_glm = roc_thresholds_glm[i]
optimal_preds_glm = np.where(glm_anova_raw_preds[:,1] < threshold_glm, 0, 1)
print(threshold_glm.round(3))
print(confusion_matrix(y_test_p, optimal_preds_glm))
print(classification_report(y_test_p, optimal_preds_glm))


fig = plt.figure(figsize = (12,6))
ax = plt.subplot(1,2,1)
ax.plot(fpr, tpr, lw=2.5, c='green', alpha=0.7)
ax.plot(fpr_glm, tpr_glm, lw=2.5, c='red', alpha=0.7)
ax.plot([0,1],[0,1], ls='--', color='black')
plt.title('ROC Curve', fontsize=20)
plt.xlabel('FPR', fontsize=18)
plt.ylabel('TPR', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(['RF AUC = {}'.format(auc.round(3)), 'GLM AUC = {}'.format(auc_glm.round(3)), 'No-Skill'], loc='lower right', fontsize=16)

ax2 = plt.subplot(1,2,2)    
ax2.plot(recalls, precisions, lw=2.5, c='green', alpha=0.7)
ax2.plot(recalls_glm, precisions_glm, lw=2.5, c='red', alpha=0.7)
ax2.plot([0,1],[recall_no_skill,recall_no_skill], ls='--', color='black')
plt.title('Precision-Recall Curve', fontsize=18)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax2.set_xlim(0,1)
ax2.set_ylim(0,1)
plt.legend(['RF', 'GLM', 'No-Skill'], loc='lower left', fontsize=16)
plt.tight_layout()
plt.show()

