
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import gc
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb

import os
os.getcwd()
os.chdir('C:/Users/Qiao Yu/Desktop/credit card default detection')

'''
# II. Data 
# 2.1 Data Overview
'''
data = pd.read_csv('creditcard.csv')
print('Credit Card Fraud Detection Data -- rows:', data.shape[0],\
      'columns:', data.shape[1])
data.head()
data.describe()
# Looking to the Time feature, we can confirm that the data contains 284,807 transactions,
# during 2 consecutive days (or 172792 seconds).

'''
# 2.2 Check missing data
'''
total = data.isnull().sum().sort_values(ascending = False)
percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending= False)
pd.concat([total, percent], axis = 1, keys= ['Total', 'Percent']).transpose()

'''
# 2.3 Check data imbalance
'''
count_classes = pd.value_counts(data['Class'], sort = True)
count_classes.head()
count_classes.plot(kind = 'bar')
plt.title('Fraud Class Histogram')
plt.xlabel('Class')
plt.ylabel('Number')
plt.show()
plt.savefig('Fraud Class')

'''
# III.  Data Feature Analysis
'''
# Transactions in time
class_0 = data.loc[data['Class'] == 0]['Time']
class_1 = data.loc[data['Class'] == 1]['Time']
hist_data = [class_0, class_1]
group_labels = ['Not Fraud', 'Fraud']

fig = ff.create_distplot(hist_data, group_labels, show_hist = False, show_rug = False)
fig['layout'].update(title = 'Credit Card Transactions Time Density Plot', xaxis = dict(title = 'Time [Secds]'))
#iplot(fig, filename = 'dist_only')
plot(fig, filename='dist_only')
# Transactions in Amount
fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12,6))
s = sns.boxplot(ax = ax1, x="Class", y="Amount", hue="Class",data=data, showfliers=False)
s = sns.boxplot(ax = ax2, x="Class", y="Amount", hue="Class",data=data, showfliers=True)
ax1.set_title("Class x Amount", fontsize=12)
ax1.set_xlabel("Fraud or Not?", fontsize=12)
ax1.set_ylabel("Amount", fontsize = 12)
ax2.set_title("Class x Amount", fontsize=12)
ax2.set_xlabel("Fraud or Not?", fontsize=12)
plt.show()

ax = sns.lmplot(y="Amount", x="Time", fit_reg=False,aspect=1.8,
                data=data, hue='Class')
plt.title("Amounts by Minutes of Frauds and Normal Transactions",fontsize=16)
plt.show()

# Feature correlation

plt.figure(figsize = (14,14))
plt.title('Credit Card Transactions features correlation plot (Pearson)')
corr = data.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Blues")
plt.show()

sns.lmplot(x='V20', y='Amount',data=data, hue='Class', fit_reg=True,scatter_kws={'s':2})
s=plt.gca()
s.set_title('Correlation between V20 and Frauds')
plt.show()
sns.lmplot(x='V7', y='Amount',data=data, hue='Class', fit_reg=True,scatter_kws={'s':2})
s=plt.gca()
s.set_title('Correlation between V7 and Frauds')
plt.show()
sns.lmplot(x='V2', y='Amount',data=data, hue='Class', fit_reg=True,scatter_kws={'s':2})
s=plt.gca()
s.set_title('Correlation between V2 and Frauds')
plt.show()
sns.lmplot(x='V5', y='Amount',data=data, hue='Class', fit_reg=True,scatter_kws={'s':2})
s=plt.gca()
s.set_title('Correlation between V5 and Frauds')
plt.show()

# Feature density plot
var = data.columns.values
class0 = data.loc[data['Class'] == 0]
class1 = data.loc[data['Class'] == 1]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(8,4, figsize = (16,32))
i = 0
for feature in var:
    i += 1
    plt.subplot(8,4,i)
    sns.kdeplot(class0[feature], bw = 0.5, label='Class = 0')
    sns.kdeplot(class1[feature], bw = 0.5, label='Class = 1')
    plt.xlabel(feature, fontsize = 12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show()



"""
II. Predictive models
2.0 Define Model Parameters
2.1 Logistic Regression
2.2 Random Forest
2.3 XGBoost
"""
target = 'Class'
predictors = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',\
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',\
       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',\
       'Amount']
# train/validation/test split
valid_size = 0.2  # simple validation
test_size = 0.2
kfolds = 5 # number of KFolds for cross-validation
random_state = 2019

train_d, test_d = train_test_split(data, test_size = 0.2, random_state = 2019, shuffle = True)
train_d, valid_d = train_test_split(train_d, test_size = 0.2, random_state = 2019, shuffle = True)

'''
2.1 Logistic Regression
'''
from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression(random_state=2019,solver='liblinear')
param_grid = {'C': [0.1, 1, 10, 20],'penalty':['l1', 'l2']}
grid_search_lr = GridSearchCV(lgr, param_grid=param_grid, scoring='recall', cv=5)
grid_search_lr.fit(train_d[predictors], train_d[target].values)
print('The best recall scores:', grid_search_lr.best_score_,\
      'Best parameter for trainning set:',grid_search_lr.best_params_)
# The best recall scores: 0.6842071628121266 Best parameter for trainning set: {'C': 1, 'penalty': 'l2'}
lgr = LogisticRegression(random_state=2019, penalty='l2',C=1,solver='liblinear')
lgr.fit(train_d[predictors], train_d[target].values)
preds = lgr.predict(valid_d[predictors])
preds_t = lgr.predict(test_d[predictors])

cm = pd.crosstab(valid_d[target].values, preds, rownames=['Fact'], colnames=['Predicted'])
fig, (ax1)=plt.subplots(ncols=1, figsize=(6,6))
sns.heatmap(cm, xticklabels=['Not Fraud','Fraud'], yticklabels=['Not Fraud', 'Fraud'],\
            annot=True, ax = ax1, linewidths=0.2,linecolor='Darkblue',cmap='Blues')
plt.title('Confusion Matrix', fotsize = 14)

# Model Evaluation
roc_auc_score(valid_d[target].values, preds)
roc_auc_score(test_d[target].values, preds_t)

target_names = ['class 0', 'class 1']
print(classification_report(test_d[target].values, preds_t, target_names=target_names))
print(classification_report(valid_d[target].values, preds, target_names=target_names))

# with cross_validation;
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lgr,train_d[predictors], train_d[target].values, cv=5, scoring='roc_auc')
cross_val_score(lgr,test_d[predictors], test_d[target].values, cv=5, scoring='roc_auc') #0.8756

'''
2.2 Random Forest
'''
RFC_METRIC = 'gini'  #validation criterion, metric used for RandomForrestClassifier
N_ESTIMATORS = 100 #number of estimators/trees used for RandomForrestClassifier
N_JOBS = 4 #number of parallel jobs used for RandomForrestClassifier

clf = RandomForestClassifier(n_jobs=4, random_state=2019, criterion='gini', n_estimators=100,verbose=False)
clf.fit(train_d[predictors],train_d[target].values)
preds = clf.predict(valid_d[predictors])
preds_t = clf.predict(test_d[predictors])

tmp = pd.DataFrame({'Feature':predictors, 'Feature importance': clf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance', ascending=False)
plt.figure(figsize = (7,4))
plt.title('Feature importance', fontsize=14)
s = sns.barplot(x='Feature', y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()

cm = pd.crosstab(valid_d[target].values, preds, rownames=['Fact'], colnames=['Predicted'])
fig, ax1=plt.subplots(ncols=1, figsize=(6,6))
sns.heatmap(cm, xticklabels=['Not Fraud','Fraud'], yticklabels=['Not Fraud', 'Fraud'],\
            annot=True, ax = ax1, linewidths=0.2,linecolor='Darkblue',cmap='Blues')
plt.title('Confusion Matrix', fotsize = 14)
# Model Evaluation

#roc_auc_score(valid_d[target].values, preds)
#print(classification_report(valid_d[target].values, preds, target_names=target_names))
roc_auc_score(test_d[target].values, preds_t)
target_names = ['class 0', 'class 1']
print(classification_report(test_d[target].values, preds_t, target_names=target_names))


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
2.3 XGBoost
# XGBoost is a gradient boosting algorithm.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# prepare the model
dtrain = xgb.DMatrix(train_d[predictors], train_d[target].values)
dvalid = xgb.DMatrix(valid_d[predictors],valid_d[target].values)
dtest = xgb.DMatrix(test_d[predictors], test_d[target].values)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

# Set XGBoost parameters
params = {}
params['objective'] = 'binary:logistic'
params['eta'] = 0.039
params['silent'] = True
params['max_depth'] = 2
params['subsample'] = 0.8
params['colsample_bytree'] = 0.9
params['eval_metric'] = 'auc'
params['random_state'] = 2019

model = xgb.train(params, dtrain, 1000, watchlist, early_stopping_rounds=50,maximize = True, verbose_eval=50)
# valid-auc: 0.979, for round 309
fig, (ax) = plt.subplots(ncols=1, figsize=(8,5))
xgb.plot_importance(model, height=0.8, title="Features importance (XGBoost)", ax=ax, color="darkblue")
plt.show()

# predict test set
preds = model.predict(dtest)
roc_auc_score(test_d[target].values, preds)
# 0.976337108511141

"""
自定义评价函数为 precision， recall， F1；Just a test! 
Tried, Failed......
Try again! And I believe success this time! And the fact is it's TRUE!!!!
"""

# labels == 0: no fraud
# labels == 1: fraud
preds = model.predict(dvalid)
labels = valid_d[target].values
def eval_boost(preds, labels):
    act_pos = sum(labels == 0)
    act_neg = labels.shape[0] - act_pos
    true_pos = sum(1 for i in range(len(preds)) if (preds[i] <= 0.5) & (labels[i] == 0))
    false_pos = sum(1 for i in range(len(preds)) if (preds[i] <= 0.5) & (labels[i] == 1))
    false_neg = act_pos - true_pos
    true_neg = act_neg - false_pos
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f_score = 2 * precision * recall / (precision + recall)

    print('\n confusion matrix')
    print('-------------------')
    print('tp:{:6d} fp:{:6d}'.format(true_pos, false_pos))
    print('fn:{:6d} tn:{:6d}'.format(false_neg, true_neg))
    print('-------------------')
    print('Precision: {:.6f}\nRecall: {:.6f}\nF1 score: {:.6f}\n'.format(precision, recall, f_score))
preds = model.predict(dvalid)
roc_auc_score(valid_d[target].values, preds)
label = labels = valid_d[target].values
eval_boost(preds, labels)

roc_auc_score(test_d[target].values, preds_t)
preds_t = model.predict(dtest)
labels_t = test_d[target].values
eval_boost(preds_t, labels_t)
'''''''''''''''''''''''''''''''''''''''''''''''''''
# 2.4 LightGBM: Another gradient boosting algorithm

'''''''''''''''''''''''''''''''''''''''''''''''''''

# Define model parameters
params = {
    'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc',
    'learning_rate': 0.03, 'num_leaves': 7, 'max_depth': 3,
    'min_child_samples':100, # min_data_in_leaf
    'max_bin': 100, #number of bucked bin for feature values
    'subsample': 0.9, # subsample ratio of the training instance
    'subsample_freq':1, # frequence of subsample
    'colsample_bytree': 0.7, # subsample ratio of columns when constructing each tree.
    'min_child_weight':0,
    'min_split_gain':0, # lambda_l1, lambda_l2 and min_gain_to_split to regularization.
    'nthread':8, 'verbose': 0, 'scale_pos_weight': 150 # because training data is extremely unbalanced
}
MAX_ROUNDS = 1000 #lgb iterations
EARLY_STOP = 50 #lgb early stop
OPT_ROUNDS = 1000  #To be adjusted based on best validation rounds
VERBOSE_EVAL = 50 #Print out metric result
IS_LOCAL = False
# prepare model
dtrain = lgb.Dataset(train_d[predictors].values,label=train_d[target].values,feature_name=predictors)
dvalid = lgb.Dataset(valid_d[predictors].values,label=valid_d[target].values,feature_name=predictors)

evals_results = {}
model = lgb.train(params, dtrain, valid_sets=[dtrain, dvalid],
                  valid_names=['train', 'valid'],
                  evals_result = evals_results, num_boost_round=1000,
                  early_stopping_rounds= 100, verbose_eval=50, feval = None)

fig, (ax) = plt.subplots(ncols=1, figsize=(8,5))
lgb.plot_importance(model, height=0.8, title="Features importance (LightGBM)", ax=ax,color="red")
plt.show()

preds = model.predict(test_d[predictors])
labels = test_d[target].values
eval_boost(preds, labels)
roc_auc_score(test_d[target].values, preds) # 0.9577174941792705


## with cross validation
kf = KFold(n_splits=5, random_state=2019, shuffle=True)
# Create arrays and dataframes to store results
oof_preds = np.zeros(train_d.shape[0])
test_preds = np.zeros(test_d.shape[0])
feature_importance_df = pd.DataFrame()
n_fold = 0
for train_idx, valid_idx in kf.split(train_d):
    train_x, train_y = train_d[predictors].iloc[train_idx], train_d[target].iloc[train_idx]
    valid_x, valid_y = train_d[predictors].iloc[valid_idx], train_d[target].iloc[valid_idx]

    evals_results = {}
    model = LGBMClassifier(
        nthread=-1,
        n_estimators=2000,
        learning_rate=0.01,
        num_leaves=80,
        colsample_bytree=0.98,
        subsample=0.78,
        reg_alpha=0.04,
        reg_lambda=0.073,
        subsample_for_bin=50,
        boosting_type='gbdt',
        is_unbalance=False,
        min_split_gain=0.025,
        min_child_weight=40,
        min_child_samples=510,
        objective='binary',
        metric='auc',
        silent=-1,
        verbose=-1,
        feval=None)
    model.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
              eval_metric='auc', verbose=VERBOSE_EVAL, early_stopping_rounds=EARLY_STOP)

    oof_preds[valid_idx] = model.predict_proba(valid_x, num_iteration=model.best_iteration_)[:, 1]
    test_preds += model.predict_proba(test_d[predictors], num_iteration=model.best_iteration_)[:, 1] / kf.n_splits

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = predictors
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = n_fold + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
    del model, train_x, train_y, valid_x, valid_y
    gc.collect()
    n_fold = n_fold + 1
train_auc_score = roc_auc_score(train_d[target], oof_preds)
print('Full AUC score %.6f' % train_auc_score)

import os
os.getcwd()
os.chdir('C:/Users/Qiao Yu/Desktop')
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
res = pd.read_csv('233.csv')
sns.set(style="whitegrid")
ax = sns.catplot(x="metrics", y="result", hue="model",kind = "bar", palette="Blues_d", size = 6,legend_out = True, data=res)
plt.title("Credit Card Fraud Detection", fontsize = 16)