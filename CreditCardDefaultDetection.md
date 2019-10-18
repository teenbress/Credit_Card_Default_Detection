##  Credit Card Fraud Detection
Qiao Yu

### Content
#### I. Introduction
#### II.Data Preparation
* 2.1 Data Overview
* 2.2 Check Missing Data
* 2.3 Check Data Balance
#### III. Data Features Analysis 
#### IV. Modelling
* 4.0 Define Model Parameters
* 4.1 Logistic Regression Model
* 4.2 Random Forest
* 4.3 XGBoost
* 4.4 lightGBM
#### V. Conclusions

### I. Introduction
The PwC global economic crime survey of 2016 suggests that approximately 36% of organizations experienced economic crime. 
Therefore, there is definitely a need to solve the problem of credit card fraud detection. 
The task of fraud detection often boils down to outlier detection, in which a dataset is scanned through to find 
potential anomalies in the data. In the past, this was done by employees  which checked all transactions manually. 
With the rise of machine learning, artificial intelligence, machine learning and other relevant fields of information technology,
it becomes feasible to automate this process and to save some of the intensive amount of labor that is put into 
detecting credit card fraud. In the following sections, my machine learning based Pythonic approach is explained.
This report contains two parts devoted to Exploratory Analysis and Prediction Models.
I'll contain the most popular four imbalanced data mining models: LR, Random Forest, XGBoost, LightGBM,
in order to find the best model which will help detect default and answer the questions:
1. How to analyze imbalanced big data?
2. Which algorithms are the strongest predicting models for default payment?  
I hope you'll enjoy this report.

### II. Data Preparation
##### Data overview
The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
The data set comes from the [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud#creditcard.csv).  

 It contains only numerical input variables which are the result of a PCA transformation.
 Unfortunately, due to confidentiality issues, we cannot provide the original features and
 more background information about the data. 
 * Features V1, V2, ... V28 are the principal components obtained with PCA, 
 the only features which have not been transformed with PCA are 'Time' and 'Amount'.
 * Time: Contains the seconds elapsed between each transaction and the first transaction 
 in the dataset. 
 * Amount: The transaction Amount, this feature can be used for example-dependant 
 cost-senstive learning. 
 * Class: The response variable and it takes value 1 in case of fraud and 0 otherwise.

###### Set Up
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

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
```
Credit Card Fraud Detection data -  rows: 284807  columns: 31
```
data = pd.read_csv('creditcard.csv')
print('Credit Card Fraud Detection Data -- rows:', data.shape[0],\
      'columns:', data.shape[1])
data.head()
data.describe()
```
The Credit Card Fraud Detection data contains 284807 rows and 31 columns, the target variable is 'Class', the other two of which are **Time** and **Amount**. For **Time**, we can confirm that the data contains 284,807 transactions, during 2 consecutive days (or 172792 seconds). The rest are output from the PCA transformation. 
##### Check Missing Data
```
data.isnull().sum().sort_values(ascending = False)
percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending= False)
```
There is no missing data. Then let's check the data balance.
##### Check Data Balance
The target variable **class** set 0 as Not fraud, and 1 as fraud.
Only 492 (or 0.172%) of transaction are fraudulent. 
That means the data is highly unbalanced with respect with target variable Class.
```
count_classes = pd.value_counts(df['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction class distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency");
```

### III. Data Features Analysis
##### Transactions in Time
```
class_0 = data.loc[data['Class'] == 0]['Time']
class_1 = data.loc[data['Class'] == 1]['Time']
hist_data = [class_0, class_1]
group_labels = ['Not Fraud', 'Fraud']

fig = ff.create_distplot(hist_data, group_labels, show_hist = False, show_rug = False)
fig['layout'].update(title = 'Credit Card Transactions Time Density Plot', xaxis = dict(title = 'Time [Secds]'))
plot(fig, filename='dist_only')
```
The plot shows:
Fraudulent transactions have a more equal distribution than actual transactions distribution, 
neither peaks nor valleys. But when a transaction happened in night, it will be more 
possible to be a fraudulent transaction.
##### Transactions in Amount
```
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
```

##### Feature density plot
```
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
```
### IV. Modelling
##### 4.0 Define Model Parameters
```
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
```
##### 4.1 Logistic Regression Model
Let's start from Logsitic regression model. I used GridSearchCV to select some parameters. It offered the best 'c'= 1, penalty = 'l2', and solver = 'liblinear'.
```
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
```
##### Model Evaluation
How to evaluate our model?  
In this case, our dataset is highly unbalanced, 
it's more important to detect the fraudulent transactions than the normal ones.
We'll use **Precision, Recall, F1 score** as our metrics, besides,
 **roc_auc_score** and the curves 
will also give good evaluation for all the models.
The area under ROC curve is **0.84**.
The 0 classes (transactions without fraud) are predicted with 100% precision 
and recall. It has some issues with detecting the 1 classes (transactions which are fraudulent). 
It can predict fraud with 71% precision. This means that 29% of the transactions which are 
fraudulent remain undetected by the system. 
The result is not bad, but can we do something to make it better?
Let's take a look at the second model : Random Forest.
##### 4.2 Random Forest

I'll use GINI as criterion, number of estimators is set to 100. We can also get feature importance rank through
this model.The most important features are **V17, V12, V14, V10, V11, V16**.
```
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
```
##### Model Evaluation
```
roc_auc_score(test_d[target].values, preds_t)
target_names = ['class 0', 'class 1']
print(classification_report(test_d[target].values, preds_t, target_names=target_names))
```
The area under ROC curve is **0.90**. Let's focus on detecting the 1 classes (transactions 
which are fraudulent). It can predict normal transactions with **89%** precision, sounds good. 
It also got a recall rate at 0.79, which means **21%** of fraudulent undetected by the system,
 which is 10% higher than logistic regression model. And its F1 score is **0.84**.
 ##### 4.3 XGBoost
 
 Random Forest is a kind of bagging algorithm, while XGBoost and LightGBM are popular boosting algorithms.
 Let's start from XGBoost.
 ```
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
```

```
# predict test set
preds = model.predict(dtest)
roc_auc_score(test_d[target].values, preds)
# 0.976337108511141
 ```
 As XGBoost only offer the ROC_AUC_Score, hence I wrote a function to evaluate the result 
 through out selected metrics, such as precision, recall and F1 score. The function could 
 also be applied on XGBoost, setting the parameter 'feval'.
 ```
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

 ```
 ```
roc_auc_score(test_d[target].values, preds_t)
preds_t = model.predict(dtest)
labels_t = test_d[target].values
eval_boost(preds_t, labels_t) 
 ```
 The AUC score for the prediction of test set is **0.976**, Its precision is **0.9996**, 
 its recall is **0.9997** and its f1 score is **0.9997**! XGBoost always gives the best 
 and almost perfect result!
 
 #### 4.4 LightGBM
 ```
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
 ```
 Train the model
 ```
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
 ```
 
 The ROC-AUC score obtained for the test set is **0.958**, lower than XGBoost, recall and f1 score also a bit lower
 than XGBoost, but much higher than logistic regression and random forest.
 
 
#### V. Conclusions

We investigated the data, checking for data imbalance, analyzing the data features and understanding the relationship 
between different features. We then investigated four predictive models. For all the four models, we used both valid 
and test data sets, the table below listed all the results obtained by test data. Overall, 
this study suggested that for the modeling of credit card risk recall rate of 0.9998 achieved 
by XGBoost model has the best performance  compared to other models, LightGBM is the second rank top algorithm for
imbalanced data.
 
 






