#!/usr/bin/env python
# coding: utf-8

# ## Project description

# <div class="paragraph">Beta Bank customers are leaving: little by little, chipping away every month. The bankers figured out it’s cheaper to save the existing customers rather than to attract new ones.</div><div class="paragraph">We need to predict whether a customer will leave the bank soon. You have the data on clients’ past behavior and termination of contracts with the bank.</div><div class="paragraph">Build a model with the maximum possible <em>F1</em> score. To pass the project, you need an <em>F1</em> score of at least 0.59. Check the <em>F1</em> for the test set.</div><div class="paragraph">Additionally, measure the <em>AUC-ROC</em> metric and compare it with the <em>F1</em>.</div></div>

# ### 1. Download and prepare the data. Explain the procedure.

# <div class="markdown markdown_size_normal markdown_type_theory"><strong>Features</strong><ul><li><em>RowNumber</em> — data string index</li><li><em>CustomerId</em> — unique customer identifier</li><li><em>Surname</em> — surname</li><li><em>CreditScore</em> — credit score</li><li><em>Geography</em> — country of residence</li><li><em>Gender</em> — gender</li><li><em>Age</em> — age</li><li><em>Tenure</em> — period of maturation for a customer’s fixed deposit (years)</li><li><em>Balance</em> — account balance</li><li><em>NumOfProducts</em> — number of banking products used by the customer</li><li><em>HasCrCard</em> — customer has a credit card</li><li><em>IsActiveMember</em> — customer’s activeness</li><li><em>EstimatedSalary</em> — estimated salary</li></ul><div class="paragraph"><strong>Target</strong></div><ul><li><em>Exited</em> — сustomer has left</li></ul></div>

# In[201]:


get_ipython().system('pip install catboost -q')


# In[202]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import *
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, LogisticRegressionCV
from sklearn.utils import shuffle
from catboost import CatBoostClassifier
import random
random.seed(42)
np.random.seed(42)
import warnings
warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


# In[203]:


def display_classification_report(y_true, y_pred):
    display(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T)


# In[204]:


def plot_roc(y_test, preds, ax=None, label='model'):
    with plt.style.context('seaborn-whitegrid'):
        if not ax: fig, ax = plt.subplots(1, 1)
        fpr, tpr, thresholds = roc_curve(y_test, preds)
        ax.plot([0, 1], [0, 1],'r--')
        ax.plot(fpr, tpr, lw=2, label=label)
        ax.legend(loc='lower right')
        ax.set_title(
             'ROC curve\n'
            f""" AP: {average_precision_score(
                y_test, preds, pos_label=1
            ):.2} | """
            f'AUC: {auc(fpr, tpr):.2}')
        ax.set_xlabel('False Positive Rate (FPR)')
        ax.set_ylabel('True Positive Rate (TPR)')
        ax.annotate(f'AUC: {auc(fpr, tpr):.2}', xy=(.43, .025))
        ax.legend()
        ax.grid()
        return ax
    

def plot_pr(y_test, preds, ax=None, label='model'):
    with plt.style.context('seaborn-whitegrid'):
        precision, recall, thresholds = precision_recall_curve(y_test, preds)
        if not ax: fig, ax = plt.subplots()
        ax.plot([0, 1], [1, 0],'r--')    
        ax.plot(recall, precision, lw=2, label=label)
        ax.legend()
        ax.set_title(
            'Precision-recall curve\n'
            f""" AP: {average_precision_score(
                y_test, preds, pos_label=1
            ):.2} | """
            f'AUC: {auc(recall, precision):.2}'
        )
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid()
        return ax


# In[205]:


#missing value ratio
def missing_values(df):
    df_nulls=pd.concat([df.dtypes, df.isna().sum(), df.isna().sum()/len(df)], axis=1)
    df_nulls.columns = ["type","count","missing_ratio"]
    df_nulls=df_nulls[df_nulls["count"]>0]
    df_nulls.sort_values(by="missing_ratio", ascending=False)
    return df_nulls

#outliers by 3 sigma rule
def outlier(data):
    data_mean, data_std = np.mean(data), np.std(data)
    cut_off = data_std * 3
    lower, upper = data_mean - cut_off, data_mean + cut_off
    outliers = [x for x in data if x < lower or x > upper]
    outliers_removed = [x for x in data if x >= lower and x <= upper]
    return len(outliers)

# full description statistics 
def describe_full(df, target_name=""):
    data_describe = df.describe().T
    df_numeric = df._get_numeric_data()
    if target_name in df.columns:
        corr_with_target=df_numeric.drop(target_name, axis=1).apply(lambda x: x.corr(df_numeric[target_name]))
        data_describe['corr_with_target']=corr_with_target
    dtype_df = df_numeric.dtypes
    data_describe['dtypes'] = dtype_df
    data_null = df_numeric.isnull().sum()/len(df) * 100
    data_describe['Missing %'] = data_null
    Cardinality = df_numeric.apply(pd.Series.nunique)
    data_describe['Cardinality'] = Cardinality
    df_skew = df_numeric.skew(axis=0, skipna=True)
    data_describe['Skew'] = df_skew
    data_describe['outliers']=[outlier(df_numeric[col]) for col in df_numeric.columns]
    data_describe['kurtosis']=df_numeric.kurtosis()
    return data_describe


# In[206]:


df = pd.read_csv('https://code.s3.yandex.net/datasets/Churn.csv')


# In[207]:


df = shuffle(df, random_state=42)


# In[208]:


features = ['CreditScore', 'Geography','Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary']
target = 'Exited'


# In[209]:


df.info()


# In[210]:


df.head()


# In[211]:


df['Gender'] = (df['Gender']=='Female').astype(int)


# In[212]:


df.isna().sum()


# In[213]:


df = df.drop(['RowNumber'], axis=1)


# In[214]:


le_geo = LabelEncoder()
df['Geography'] = le_geo.fit_transform(df['Geography'])


# In[215]:


len(df['Surname'].unique())


# In[216]:


df = df.drop(['CustomerId', 'Surname'], axis=1)


# In[217]:


df[df['Tenure'].isna()]


# In[218]:


df[df['Tenure'].isna()][target].value_counts()


# as we saw , we have missing data at the 'tenure' column, around 10%, in my opinion, it's a lot for dropping, especially if we have positive class value in imbalancing case

# let's impute it, using linear regression (without label for avoiding leakage)

# In[219]:


parameters = set(df.columns) - set([target, 'Tenure'])
test_data = df[df['Tenure'].isna()]
train_data = df[~df['Tenure'].isna()]
X = train_data[parameters].values
y = train_data['Tenure'].values
X_test, y_test = test_data[parameters].values, test_data['Tenure'].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
lr = LinearRegression().fit(X_train, y_train)
y_pred = lr.predict(X_val).astype(int)
mean_squared_error(y_val, y_pred)


# In[220]:


df.loc[df['Tenure'].isna(), 'Tenure'] = lr.predict(X_test).astype(int)


# In[221]:


df['Tenure'] = df['Tenure'].astype(int)


# In[222]:


describe_full(df, target)


# In[223]:


sns.boxplot(df['EstimatedSalary'])


# In[224]:


sns.boxplot(df['Balance'])


# In[225]:


sns.distplot(df['Balance'])


# Let's see most important features

# In[226]:


X, y = df[features].values,df[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
rfc = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = rfc.predict(X_test)
df_feature_importances = pd.DataFrame(((zip(features, rfc.feature_importances_)))).rename(columns={0:"feature",1:"coeff"}).sort_values(by="coeff", ascending = False )
sns.barplot(data=df_feature_importances, x=df_feature_importances["coeff"], y=df_feature_importances["feature"])


# features sorted by importance

# In[192]:


features = list(df_feature_importances["feature"])
features


# In[193]:


del X, y, X_test, y_test, X_train, y_train, X_val, y_val


# ### Summary

# Our data consists from 14 features, where we found out that RowNumber, CustomerId and Surname are not important (don't impact on the model) features by importance: 'Age','EstimatedSalary','Balance','CreditScore','NumOfProducts',
#  'Tenure','IsActiveMember','Geography','Gender','HasCrCard'.
#  Our target is 'Exited'. Also we have Balance, EstimatedSalary and good questiong what is meause of those columns(we could infer it from column of Geography, if it's Europe probably it's euro). Also some data was missed at Tenure column, but using some simplistic model (linear regression) we imputed it. 

# ### 2. Examine the balance of classes. Train the model without taking into account the imbalance. Briefly describe your findings.

# In[194]:


sns.countplot(df[target])


# In[195]:


display(df[target].value_counts())
100*df[target].value_counts()/len(df)


# as we see classes are imbalanced

# #### Logistic regresion

# In[196]:


history = dict()


# In[197]:


X, y = df[features].values, df[target].values


# In[198]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[199]:


lr = LogisticRegressionCV(cv=5, random_state=42).fit(X_train, y_train)
y_pred = lr.predict(X_test)


# In[200]:


display_classification_report(y_test, y_pred)


# In[227]:


#as we saw in the table, for f1 we got very bad result
history['linear_imbalanced'] = f1_score(y_test, y_pred)
f1_score(y_test, y_pred)


# In[228]:


fig, ax = plt.subplots(1,2, figsize=(8,4))
ax = ax.flatten()
plot_pr(y_test, y_pred, ax=ax[0],label="LogisticRegressionCV")
plot_roc(y_test, y_pred, ax=ax[1],label="LogisticRegressionCV")


# In[229]:


lrcv = LogisticRegressionCV(
    Cs=[0.1,1,10], penalty='l2', tol=1e-10, scoring='neg_log_loss', cv=5,
    solver='liblinear', n_jobs=4, verbose=0, refit=True,
    max_iter=100,
).fit(X_train, y_train)
y_pred = lrcv.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
display_classification_report(y_test, y_pred)

_, axs = plt.subplots(1, 2,figsize=(10,5))
axs = axs.ravel()
plot_pr(y_test, y_pred, ax=axs[0], label="LogisticRegressionCV")
plot_roc(y_test, y_pred, ax=axs[1], label="LogisticRegressionCV")


# without dealing with imbalancing , we've got very poor model, the same, more or less than baseline, random model(i would like worser, because using random guessing according with fractions weights, random model would be better)

# #### DecisionTreeClassifier

# In[230]:


params = {
"criterion":["gini", "entropy"],
"max_depth":[2,4,8,16],
"min_samples_split":[2,4,8, 16],
"min_samples_leaf":[2,4,6]}
clf = GridSearchCV(DecisionTreeClassifier(), params, cv=5).fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(clf.score(X_train, y_train))
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
display_classification_report(y_test, y_pred)


# In[231]:


f1_score(y_test, y_pred)


# In[232]:


dt = DecisionTreeClassifier(**clf.best_params_).fit(X_train, y_train)
y_pred = dt.predict(X_test)
print(dt.score(X_train, y_train))
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
display_classification_report(y_test, y_pred)

_, axs = plt.subplots(1, 2,figsize=(10,5))
axs = axs.ravel()
plot_pr(y_test, y_pred, ax=axs[0], label="DecisionTreeClassifier")
plot_roc(y_test, y_pred, ax=axs[1], label="DecisionTreeClassifier")


# In[233]:


history['desicion_tree_imbalanced'] = f1_score(y_test, y_pred)
f1_score(y_test, y_pred)


# #### CatBoostClassifier

# In[234]:


cb = CatBoostClassifier(verbose=0, random_state=42).fit(X_train, y_train)
y_pred = cb.predict(X_test)
print(cb.score(X_train, y_train))
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
display_classification_report(y_test, y_pred)

_, axs = plt.subplots(1, 2,figsize=(10,5))
axs = axs.ravel()
plot_pr(y_test, y_pred, ax=axs[0], label="CatBoostClassifier")
plot_roc(y_test, y_pred, ax=axs[1], label="CatBoostClassifier")


# In[235]:


history['catboost_imbalanced'] = f1_score(y_test, y_pred)
f1_score(y_test, y_pred)


# ### 3. Improve the quality of the model. Make sure you use at least two approaches to fixing class imbalance. Use the training set to pick the best parameters. Train different models on training and validation sets. Find the best one. Briefly describe your findings.

# ### Linear regression

# In[236]:


lrcv = LogisticRegressionCV(
    Cs=[0.1,1,10], class_weight='balanced', penalty='l2', tol=1e-10, scoring='neg_log_loss', cv=5,
    solver='liblinear', n_jobs=4, verbose=0, refit=True,
    max_iter=100,
).fit(X_train, y_train)
y_pred = lrcv.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
display_classification_report(y_test, y_pred)

_, axs = plt.subplots(1, 2,figsize=(10,5))
axs = axs.ravel()
plot_pr(y_test, y_pred, ax=axs[0], label="LogisticRegressionCV")
plot_roc(y_test, y_pred, ax=axs[1], label="LogisticRegressionCV")


# In[237]:


history['linear'] = f1_score(y_test, y_pred)
f1_score(y_test, y_pred)


# we got much better result compare with previous result of the f1 metric, without balancing

# ### DecisionTreeClassifier

# In[238]:


params = {
"class_weight":['balanced'],
"criterion":["gini", "entropy"],
"max_depth":[2,4,8,16],
"min_samples_split":[2,4,8, 16],
"min_samples_leaf":[2,4,6]}
clf = GridSearchCV(DecisionTreeClassifier(), params, cv=5).fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(clf.score(X_train, y_train))
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
display_classification_report(y_test, y_pred)


# In[239]:


dt = DecisionTreeClassifier(**clf.best_params_).fit(X_train, y_train)
y_pred = dt.predict(X_test)
print(dt.score(X_train, y_train))
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
display_classification_report(y_test, y_pred)

_, axs = plt.subplots(1, 2,figsize=(10,5))
axs = axs.ravel()
plot_pr(y_test, y_pred, ax=axs[0], label="DecisionTreeClassifier")
plot_roc(y_test, y_pred, ax=axs[1], label="DecisionTreeClassifier")


# In[240]:


history['desicion_tree'] = f1_score(y_test, y_pred)
f1_score(y_test, y_pred)


# ### CatBoost

# In[241]:


scale_pos_weight=(y_train==0).sum()/(y_train==1).sum()


# In[242]:


cb = CatBoostClassifier(verbose=0, scale_pos_weight=scale_pos_weight,random_state=42).fit(X_train, y_train)
y_pred = cb.predict(X_test)
print(cb.score(X_train, y_train))
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
display_classification_report(y_test, y_pred)

_, axs = plt.subplots(1, 2,figsize=(10,5))
axs = axs.ravel()
plot_pr(y_test, y_pred, ax=axs[0], label="CatBoostClassifier")
plot_roc(y_test, y_pred, ax=axs[1], label="CatBoostClassifier")


# In[243]:


history['catboost'] = f1_score(y_test, y_pred)
f1_score(y_test, y_pred)


# In[244]:


history


# as we could see, we improved f1 metrics using weigths balancing technics

# ## Oversampling

# we also could use imblearn package but let's do it from scratch

# In[245]:


def oversampling(features, target, repeat):
  features_zeros = features[target == 0]
  features_ones = features[target == 1]
  target_zeros = target[target == 0]
  target_ones = target[target == 1]
  features_resampled = features_zeros
  target_resampled = target_zeros
  for i in range(repeat):
    features_resampled = np.concatenate((features_resampled, features_ones), axis=0)
    target_resampled = np.concatenate((target_resampled, target_ones), axis=0)

  features_resampled, target_resampled = shuffle(features_resampled, target_resampled, random_state=42)
  return features_resampled, target_resampled


# In[246]:


print((y_train==0).sum())
print((y_train==1).sum())
print((y_train==0).sum()//(y_train==1).sum())


# In[247]:


X_resampled, y_resampled = oversampling(X_train, y_train, repeat=4)


# In[248]:


print((y_resampled==0).sum())
print((y_resampled==1).sum())
print((y_resampled==0).sum()//(y_resampled==1).sum())


# ### Logisticregression

# In[249]:


lrcv = LogisticRegressionCV(
    Cs=[0.1,1,10], class_weight='balanced', penalty='l2', tol=1e-10, scoring='neg_log_loss', cv=5,
    solver='liblinear', n_jobs=4, verbose=0, refit=True,
    max_iter=100,
).fit(X_resampled, y_resampled)
y_pred = lrcv.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
display_classification_report(y_test, y_pred)

_, axs = plt.subplots(1, 2,figsize=(10,5))
axs = axs.ravel()
plot_pr(y_test, y_pred, ax=axs[0], label="LogisticRegressionCV")
plot_roc(y_test, y_pred, ax=axs[1], label="LogisticRegressionCV")


# In[250]:


print(f'without balancing={history["linear_imbalanced"]:.2f}')
print(f'with balancing={history["linear"]:.2f}')
print(f'using oversampling={f1_score(y_test, y_pred):.2f}')


# ### DecisionTreeClassifier

# In[251]:


params = {
"criterion":["gini", "entropy"],
"max_depth":[2,4,8,16],
"min_samples_split":[2,4,8, 16],
"min_samples_leaf":[2,4,6]}
clf = GridSearchCV(DecisionTreeClassifier(), params, cv=5).fit(X_resampled, y_resampled)
y_pred = clf.predict(X_test)
print(clf.score(X_resampled, y_resampled))
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
display_classification_report(y_test, y_pred)


# In[252]:


dt = DecisionTreeClassifier(**clf.best_params_).fit(X_resampled, y_resampled)
y_pred = dt.predict(X_test)
print(dt.score(X_resampled, y_resampled))
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
display_classification_report(y_test, y_pred)

_, axs = plt.subplots(1, 2,figsize=(10,5))
axs = axs.ravel()
plot_pr(y_test, y_pred, ax=axs[0], label="DecisionTreeClassifier")
plot_roc(y_test, y_pred, ax=axs[1], label="DecisionTreeClassifier")


# In[253]:


print(f'without balancing={history["desicion_tree_imbalanced"]:.2f}')
print(f'with balancing={history["desicion_tree"]:.2f}')
print(f'using oversampling={f1_score(y_test, y_pred):.2f}')


# ### CatBoost
# 

# In[254]:


cb = CatBoostClassifier(verbose=0, random_state=42).fit(X_resampled, y_resampled)
y_pred = cb.predict(X_test)
print(cb.score(X_resampled, y_resampled))
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
display_classification_report(y_test, y_pred)

_, axs = plt.subplots(1, 2,figsize=(10,5))
axs = axs.ravel()
plot_pr(y_test, y_pred, ax=axs[0], label="CatBoostClassifier")
plot_roc(y_test, y_pred, ax=axs[1], label="CatBoostClassifier")


# In[255]:


print(f'without balancing={history["catboost_imbalanced"]:.2f}')
print(f'with balancing={history["catboost"]:.2f}')
print(f'using oversampling={f1_score(y_test, y_pred):.2f}')


# also we could use undersampling

# ### 4. Perform the final testing.

# best result we've got using CatBoost, using class weights balancing

# In[256]:


cb = CatBoostClassifier(verbose=0, scale_pos_weight=scale_pos_weight,random_state=42).fit(X_train, y_train)
y_pred = cb.predict(X_test)
print(cb.score(X_train, y_train))
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
display_classification_report(y_test, y_pred)

_, axs = plt.subplots(1, 2,figsize=(10,5))
axs = axs.ravel()
plot_pr(y_test, y_pred, ax=axs[0], label="CatBoostClassifier")
plot_roc(y_test, y_pred, ax=axs[1], label="CatBoostClassifier")


# <div class="markdown markdown_size_normal markdown_type_theory"><h1>Project evaluation</h1><div class="paragraph">We’ve put together the evaluation criteria for the project. Read this carefully before moving on to the task.</div><div class="paragraph">Here’s what the reviewers will look at when reviewing your project:</div><ul><li>How did you prepare the data for training? Have you processed all of the feature types?</li><li>Have you explained the preprocessing steps well enough?</li><li>How did you investigate the balance of classes?</li><li>Did you study the model without taking into account the imbalance of classes?</li><li>What are your findings about the task research?</li><li>Have you correctly split the data into sets?</li><li>How have you worked with the imbalance of classes?</li><li>Did you use at least two techniques for imbalance fixing?</li><li>Have you performed the model training, validation, and final testing correctly?</li><li>How high is your <em>F1</em> score?</li><li>Did you examine the <em>AUC-ROC</em> values?</li><li>Have you kept to the project structure and kept the code neat?</li></ul></div>
