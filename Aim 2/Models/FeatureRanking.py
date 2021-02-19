# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 20:29:06 2021

@author: mainswo3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import sys
sys.path.insert(0, "S:\Dehydration_stroke\Team Emerald\Working GitHub Directories\Michael\stroke-hemodynamics\Aim 2\Models")
import Classifiers

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt




##############################
######## Load Dataset ########

# path = 'S:\Dehydration_stroke\Team Emerald\Working Data\Preprocessed\Working\Complete.csv'
path = 'S:\Dehydration_stroke\Team Emerald\Working Data\Preprocessed\Working\Complete_updated.csv'
df = pd.read_csv(path)
df = df.drop('Unnamed: 0', axis=1)

# Create X and y varables, remove label from X data
y = df['LOS']
X = df.drop(['LOS', 'mrn_csn_pair'], axis=1)

print(X.shape)
print(y.shape)


# Run a train test split on the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train_p = preprocessing.scale(X_train)
X_test_p = preprocessing.scale(X_test)

y_train_p = np.squeeze(y_train.to_numpy())
y_test_p = np.squeeze(y_test.to_numpy())




##############################
# Optimized hyperparameters already inserted
'''
######## GLM Hyperparameter Tuning ########
hyperparameter_glm = dict()
hyperparameter_glm['C'] = [0.001,0.01,0.1,1,10,100]
print(hyperparameter_glm)

logistic = LogisticRegression(solver = 'lbfgs', max_iter=200, penalty='l2')
randomizedsearch = RandomizedSearchCV(logistic, hyperparameter_glm, cv=5)
best_model_random = randomizedsearch.fit(X_train, y_train)
print(best_model_random.best_params_)
print(best_model_random.best_estimator_)
print(best_model_random.best_score_)
'''

'''
######## RF Hyperparameter Tuning ########
hyperparameter_rf = dict()
hyperparameter_rf['min_samples_split'] = [1.0,2,3,4]
hyperparameter_rf['min_samples_leaf'] = [0.1,0.3,0.4,1]
hyperparameter_rf['max_features'] = ['auto',1.0,2,3,4]
print(hyperparameter_rf)

rf = RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=None)
randomizedsearch = RandomizedSearchCV(rf, hyperparameter_rf, cv=5)
best_model = randomizedsearch.fit(X_train, y_train)
print(best_model.best_params_)
print(best_model.best_estimator_)
print(best_model.best_score_)
'''

'''
######## XGB Hyperparameter Tuning ########
hyperparameter_xg = dict()
hyperparameter_xg['booster'] = ['gbtree', 'gblinear', 'dart']
hyperparameter_xg['eta'] = [0.1,0.3,0.5]
hyperparameter_xg['max_depth'] = [4,6,8]
print(hyperparameter_xg)

xg = XGBClassifier(n_estimators=100,criterion='gini',max_depth=None)
randomizedsearch = RandomizedSearchCV(xg, hyperparameter_xg)
best_model_xg = randomizedsearch.fit(X_train, y_train)
print(best_model_xg.best_params_)
print(best_model_xg.best_estimator_)
print(best_model_xg.best_score_)
'''




########################################
######## Print Original Results ########
rf_anova = Classifiers.RandomForestModel(params = {
            'n_estimators' : 100,
            'criterion' : 'gini',
            'max_depth' : None,
            'min_samples_split' : 3,
            'min_samples_leaf' : 1,
            'max_features' : 2,
            'verbose' : 0}
    )
rf_anova.fit(X_train_p, y_train_p)
rf_anova_raw_preds, rf_anova_preds, rf_anova_score = rf_anova.predict(X_test_p, y_test_p)

fpr, tpr, roc_thresholds = roc_curve(y_test_p, rf_anova_raw_preds[:,1])
auc = roc_auc_score(y_test_p, rf_anova_raw_preds[:,1])
precisions, recalls, _ = precision_recall_curve(y_test_p, rf_anova_raw_preds[:,1])
print(confusion_matrix(y_test_p, rf_anova_preds))
print(classification_report(y_test_p, rf_anova_preds))


glm_anova = Classifiers.LogisticRegressionModel(params = {
            'penalty' : 'l2',
            'solver' : 'lbfgs',
            'max_iter' : 200,
            'C' : 0.001,
            'verbose' : 0}
    )
glm_anova.fit(X_train_p, y_train_p)
glm_anova_raw_preds, glm_anova_preds, glm_anova_score = glm_anova.predict(X_test_p, y_test_p)

fpr_glm, tpr_glm, roc_thresholds_glm = roc_curve(y_test_p, glm_anova_raw_preds[:,1])
auc_glm = roc_auc_score(y_test_p, glm_anova_raw_preds[:,1])
precisions_glm, recalls_glm, _ = precision_recall_curve(y_test_p, glm_anova_raw_preds[:,1])
print(confusion_matrix(y_test_p, glm_anova_preds))
print(classification_report(y_test_p, glm_anova_preds))


xgb = Classifiers.XGBoostModel(params = {
            'booster' : 'gbtree',
            'eta' : 0.3,
            'max_depth' : 6}
    )
xgb.fit(X_train_p, y_train_p)
xgb_raw_preds, xgb_preds, xgb_score = xgb.predict(X_test_p, y_test_p)

fpr_xgb, tpr_xgb, roc_thresholds_xgb = roc_curve(y_test_p, xgb_raw_preds[:,1])
auc_xgb = roc_auc_score(y_test_p, xgb_raw_preds[:,1])
precisions_xgb, recalls_xgb, _ = precision_recall_curve(y_test_p, xgb_raw_preds[:,1])
print(confusion_matrix(y_test_p, xgb_preds))
print(classification_report(y_test_p, xgb_preds))


recall_no_skill = y_train_p.sum()/len(y_train_p)

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

min_dist_xgb = np.inf
threshold_xgb = 0
for i in range(len(fpr_xgb)):
    dist_xgb = np.sqrt((1-tpr_xgb[i])**2 + fpr_xgb[i]**2)
    if dist_xgb < min_dist_xgb:
        min_dist_xgb = dist_xgb
        threshold_xgb = roc_thresholds_xgb[i]
optimal_preds_xgb = np.where(xgb_raw_preds[:,1] < threshold_xgb, 0, 1)
print(threshold_xgb.round(3))
print(confusion_matrix(y_test_p, optimal_preds_xgb))
print(classification_report(y_test_p, optimal_preds_xgb))


fig = plt.figure(figsize = (12,6))
ax = plt.subplot(1,2,1)
ax.plot(fpr, tpr, lw=2.5, c='green', alpha=0.7)
ax.plot(fpr_glm, tpr_glm, lw=2.5, c='red', alpha=0.7)
ax.plot(fpr_xgb, tpr_xgb, lw=2.5, c='blue', alpha=0.7)
ax.plot([0,1],[0,1], ls='--', color='black')
plt.title('ROC Curve', fontsize=20)
plt.xlabel('FPR', fontsize=18)
plt.ylabel('TPR', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(['RF AUC = {}'.format(auc.round(3)), 'GLM AUC = {}'.format(auc_glm.round(3)), 
            'XGB AUC = {}'.format(auc_xgb.round(3)), 'No-Skill'], loc='lower right', fontsize=16)

ax2 = plt.subplot(1,2,2)    
ax2.plot(recalls, precisions, lw=2.5, c='green', alpha=0.7)
ax2.plot(recalls_glm, precisions_glm, lw=2.5, c='red', alpha=0.7)
ax2.plot(recalls_xgb, precisions_xgb, lw=2.5, c='blue', alpha=0.7)
ax2.plot([0,1],[recall_no_skill,recall_no_skill], ls='--', color='black')
plt.title('Precision-Recall Curve', fontsize=18)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax2.set_xlim(0,1)
ax2.set_ylim(0,1)
plt.legend(['RF', 'GLM', 'XGB', 'No-Skill'], loc='lower left', fontsize=16)
plt.tight_layout()
plt.show()
#fig.savefig('S:/Dehydration_stroke/Team Emerald/scripts/Michael Working Scripts/Michael Figures/AllFeaturesHyperparameter.png')




########################################
##### Feature Ranking using Lasso #####
# source: kaggle.com user: dkim1992

# Run to determine optimal C value
'''
param_grid = {'logisticregression__C': [0.001,0.01,0.1,1,10,100]}
pipe = make_pipeline(StandardScaler(), LogisticRegression(penalty = 'l1', solver='liblinear'))
grid = GridSearchCV(pipe, param_grid, cv=10)
grid.fit(X_train,y_train)
print(grid.best_params_)
'''

X_scaled = StandardScaler().fit_transform(X_train)
clf = LogisticRegression(penalty = 'l1', C=0.1, solver='liblinear', max_iter=200)
clf.fit(X_scaled,y_train)

num_features = X_train.shape[1]
zero_feat = []
nonzero_feat = []
# type(clf.coef_)
for i in range(num_features):
    coef = clf.coef_[0,i]
    if coef == 0:
        zero_feat.append(X_train.columns[i])
    else:
        nonzero_feat.append((coef, X_train.columns[i]))

print('\n\n\n')
nznew = sorted(nonzero_feat, reverse = True)
print(len(nonzero_feat))
print('\n\n\n')


count_glm = {'temp':0,'pulse':0,'iv':0,'dbp':0,'sbp':0,'urine':0,'other':0}
for i in count_glm:
    print(i)
for s in nznew[:100]:
    for category in count_glm:
        if category in s[1]:
            count_glm[category] += 1
count_glm['other'] = 0
count_glm['other'] = 100 - sum(count_glm.values())

plt.bar(count_glm.keys(), count_glm.values())
plt.ylabel('count')
plt.title('GLM 100 Top Ranked Features')


lassodf = pd.DataFrame()

lassodf['Feature'] = [i[1] for i in nznew[:5]]
lassodf['Lasso Coefficient'] = [i[0] for i in nznew[:5]]
print(lassodf)
print('\n\n')

features_used = [i[1] for i in nznew]
X_glm_train = X_train[features_used]
X_glm_test = X_test[features_used]
  

# Scale and normalize raw data
X_train_new = preprocessing.scale(X_glm_train)
X_test_new = preprocessing.scale(X_glm_test)
y_train_p = np.squeeze(y_train.to_numpy())
y_test_p = np.squeeze(y_test.to_numpy())  



# Optimized hyperparameters already inserted
'''
######## GLM Hyperparameter Tuning ########
hyperparameter_glm = dict()
hyperparameter_glm['C'] = [0.001,0.01,0.1,1,10,100]
print(hyperparameter_glm)

logistic = LogisticRegression(solver = 'lbfgs', max_iter=200, penalty='l2')
randomizedsearch = RandomizedSearchCV(logistic, hyperparameter_glm, cv=5)
best_model_random = randomizedsearch.fit(X_train_new, y_train_p)
print(best_model_random.best_params_)
print(best_model_random.best_estimator_)
print(best_model_random.best_score_)
'''



glm_anova = Classifiers.LogisticRegressionModel(params = {
            'penalty' : 'l2',
            'solver' : 'lbfgs',
            'max_iter' : 200,
            'C' : 0.01,
            'verbose' : 0}
    )
glm_anova.fit(X_train_new, y_train_p)
glm_anova_raw_preds, glm_anova_preds, glm_anova_score = glm_anova.predict(X_test_new, y_test_p)

fpr_glm, tpr_glm, roc_thresholds_glm = roc_curve(y_test_p, glm_anova_raw_preds[:,1])
auc_glm = roc_auc_score(y_test_p, glm_anova_raw_preds[:,1])
precisions_glm, recalls_glm, _ = precision_recall_curve(y_test_p, glm_anova_raw_preds[:,1])
print(confusion_matrix(y_test_p, glm_anova_preds))
print(classification_report(y_test_p, glm_anova_preds))

recall_no_skill = y_train_p.sum()/len(y_train_p)

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
ax.plot(fpr_glm, tpr_glm, lw=2.5, c='red', alpha=0.7)
ax.plot([0,1],[0,1], ls='--', color='black')
plt.title('ROC Curve', fontsize=20)
plt.xlabel('FPR', fontsize=18)
plt.ylabel('TPR', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(['GLM AUC = {}'.format(auc_glm.round(3)),
            'No-Skill'], loc='lower right', fontsize=16)

ax2 = plt.subplot(1,2,2)    
ax2.plot(recalls_glm, precisions_glm, lw=2.5, c='red', alpha=0.7)
ax2.plot([0,1],[recall_no_skill,recall_no_skill], ls='--', color='black')
plt.title('Precision-Recall Curve', fontsize=18)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax2.set_xlim(0,1)
ax2.set_ylim(0,1)
plt.legend(['GLM', 'No-Skill'], loc='lower left', fontsize=16)
plt.tight_layout()
plt.show()
#fig.savefig('S:/Dehydration_stroke/Team Emerald/scripts/Michael Working Scripts/Michael Figures/EPI1Full.png')




##### Feature Ranking using RF #####

# Rank all features in the training set
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

feat_labels = X_train.columns

clf = RandomForestClassifier(n_estimators=100, criterion='gini',
                             max_depth=None, min_samples_split=3,
                             min_samples_leaf=1, max_features=2)
clf.fit(X_train, y_train)
print(X_train.shape)
print(y_train.shape)

rf_features = []
for feature in zip(feat_labels, clf.feature_importances_):
    rf_features.append(feature)

newdf = pd.DataFrame()

new = sorted(rf_features, key=lambda x:x[1], reverse=True)
newdf['Feature'] = [i[0] for i in new[:5]]
newdf['RF Score'] = [i[1] for i in new[:5]]
print(newdf)
print('\n\n')


count_rf = {'temp':0,'pulse':0,'iv':0,'dbp':0,'sbp':0,'urine':0,'other':0}
for i in count_rf:
    print(i)
for s in new[:100]:
    for category in count_rf:
        if category in s[0]:
            count_rf[category] += 1
count_rf['other'] = 0
count_rf['other'] = 100 - sum(count_rf.values())
print(count_rf)

plt.bar(count_rf.keys(), count_rf.values())
plt.ylabel('count')
plt.title('RF 100 Top Ranked Features')


# Retrain model with only top features
sfm = SelectFromModel(clf, threshold = 0.0005)
sfm.fit(X_train, y_train)

#for feature_list_index in sfm.get_support(indices=True):
#    print(feat_labels[feature_list_index])
    
X_train_new = sfm.transform(X_train)
X_test_new = sfm.transform(X_test)
print(X_train_new)
print(X_train_new.shape)
clf_new = RandomForestClassifier(n_estimators=100, random_state=0)
clf_new.fit(X_train_new, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))


y_pred_new = clf_new.predict(X_test_new)
print(accuracy_score(y_test, y_pred_new))
 

# Scale and normalize raw data
X_train_p = preprocessing.scale(X_train_new)
X_test_p = preprocessing.scale(X_test_new)
y_train_p = np.squeeze(y_train.to_numpy())
y_test_p = np.squeeze(y_test.to_numpy())  



# Optimized hyperparameters already inserted
'''
######## RF Hyperparameter Tuning ########
hyperparameter_rf = dict()
hyperparameter_rf['min_samples_split'] = [1.0,2,3,4]
hyperparameter_rf['min_samples_leaf'] = [0.1,0.3,0.4,1]
hyperparameter_rf['max_features'] = ['auto',1.0,2,3,4]
print(hyperparameter_rf)

rf = RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=None)
randomizedsearch = RandomizedSearchCV(rf, hyperparameter_rf, cv=5)
best_model = randomizedsearch.fit(X_train_new, y_train_p)
print(best_model.best_params_)
print(best_model.best_estimator_)
print(best_model.best_score_)

######## XGB Hyperparameter Tuning ########
hyperparameter_xg = dict()
hyperparameter_xg['booster'] = ['gbtree', 'gblinear', 'dart']
hyperparameter_xg['eta'] = [0.1,0.3,0.5]
hyperparameter_xg['max_depth'] = [4,6,8]
print(hyperparameter_xg)

xg = XGBClassifier(n_estimators=100,criterion='gini',max_depth=None)
randomizedsearch = RandomizedSearchCV(xg, hyperparameter_xg)
best_model_xg = randomizedsearch.fit(X_train_new, y_train_p)
print(best_model_xg.best_params_)
print(best_model_xg.best_estimator_)
print(best_model_xg.best_score_)
'''



rf_anova = Classifiers.RandomForestModel(params = {
            'n_estimators' : 100,
            'criterion' : 'gini',
            'max_depth' : None,
            'min_samples_split' : 3,
            'min_samples_leaf' : 1,
            'max_features' : 3,
            'verbose' : 0}
    )
rf_anova.fit(X_train_p, y_train_p)
rf_anova_raw_preds, rf_anova_preds, rf_anova_score = rf_anova.predict(X_test_p, y_test_p)

fpr, tpr, roc_thresholds = roc_curve(y_test_p, rf_anova_raw_preds[:,1])
auc = roc_auc_score(y_test_p, rf_anova_raw_preds[:,1])
precisions, recalls, _ = precision_recall_curve(y_test_p, rf_anova_raw_preds[:,1])
print(confusion_matrix(y_test_p, rf_anova_preds))
print(classification_report(y_test_p, rf_anova_preds))


xgb = Classifiers.XGBoostModel(params = {
            'booster' : 'gbtree',
            'eta' : 0.1,
            'max_depth' : 4}
        )
xgb.fit(X_train_p, y_train_p)
xgb_raw_preds, xgb_preds, xgb_score = xgb.predict(X_test_p, y_test_p)

fpr_xgb, tpr_xgb, roc_thresholds_xgb = roc_curve(y_test_p, xgb_raw_preds[:,1])
auc_xgb = roc_auc_score(y_test_p, xgb_raw_preds[:,1])
precisions_xgb, recalls_xgb, _ = precision_recall_curve(y_test_p, xgb_raw_preds[:,1])
print(confusion_matrix(y_test_p, xgb_preds))
print(classification_report(y_test_p, xgb_preds))


recall_no_skill = y_train_p.sum()/len(y_train_p)


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

min_dist_xgb = np.inf
threshold_xgb = 0
for i in range(len(fpr_xgb)):
    dist_xgb = np.sqrt((1-tpr_xgb[i])**2 + fpr_xgb[i]**2)
    if dist_xgb < min_dist_xgb:
        min_dist_xgb = dist_xgb
        threshold_xgb = roc_thresholds_xgb[i]
optimal_preds_xgb = np.where(xgb_raw_preds[:,1] < threshold_xgb, 0, 1)
print(threshold_xgb.round(3))
print(confusion_matrix(y_test_p, optimal_preds_xgb))
print(classification_report(y_test_p, optimal_preds_xgb))


fig = plt.figure(figsize = (12,6))
ax = plt.subplot(1,2,1)
ax.plot(fpr, tpr, lw=2.5, c='green', alpha=0.7)
ax.plot(fpr_xgb, tpr_xgb, lw=2.5, c='blue', alpha=0.7)
ax.plot([0,1],[0,1], ls='--', color='black')
plt.title('ROC Curve', fontsize=20)
plt.xlabel('FPR', fontsize=18)
plt.ylabel('TPR', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(['RF AUC = {}'.format(auc.round(3)), 'XGB AUC = {}'.format(auc_xgb.round(3)),
            'No-Skill'], loc='lower right', fontsize=16)

ax2 = plt.subplot(1,2,2)    
ax2.plot(recalls, precisions, lw=2.5, c='green', alpha=0.7)
ax2.plot(recalls_xgb, precisions_xgb, lw=2.5, c='blue', alpha=0.7)
ax2.plot([0,1],[recall_no_skill,recall_no_skill], ls='--', color='black')
plt.title('Precision-Recall Curve', fontsize=18)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax2.set_xlim(0,1)
ax2.set_ylim(0,1)
plt.legend(['RF', 'XGB', 'No-Skill'], loc='lower left', fontsize=16)
plt.tight_layout()
plt.show()
#fig.savefig('S:/Dehydration_stroke/Team Emerald/scripts/Michael Working Scripts/Michael Figures/EPI1Full.png')







