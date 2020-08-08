#!/usr/bin/env python
# coding: utf-8

# ### Project description

# <div class="paragraph">Mobile carrier Megaline has found out that many of their subscribers use legacy plans. They want to develop a model that would analyze subscribers' behavior and recommend one of Megaline's newer plans: Smart or Ultra. </div><div class="paragraph">You have access to behavior data about subscribers who have already switched to the new plans (from the project for the Statistical Data Analysis course). For this classification task, you need to develop a model that will pick the right plan. Since you’ve already performed the data preprocessing step, you can move straight to creating the model.  </div><div class="paragraph">Develop a model with the highest possible <em>accuracy</em>. In this project, the threshold for <em>accuracy</em> is 0.75. Check the <em>accuracy</em> using the test dataset.  </div>

# ### Data description

# <div class="paragraph">Every observation in the dataset contains monthly behavior information about one user. The information given is as follows: </div>
# <ul><li><em>сalls</em> — number of calls,</li><li><em>minutes</em> — total call duration in minutes,</li><li><em>messages</em> — number of text messages,</li><li><em>mb_used</em> — Internet traffic used in MB,</li><li><em>is_ultra</em> — plan for the current month (Ultra - 1, Smart - 0).</li></ul>

# ### Step 1.
# <br>
# <li>Open and look through the data file. Path to the file:<code class="code-inline code-inline_theme_light">datasets/users_behavior.csv</code></li>

# In[49]:


get_ipython().system('pip install imblearn -q')


# In[92]:


get_ipython().system('pip install catboost -q')


# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, RandomizedSearchCV, ParameterGrid
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import *
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
get_ipython().run_line_magic('matplotlib', 'inline')
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore")
from sklearn import set_config
set_config(print_changed_only=False)


# In[3]:


## global seed ##
import random
rnd_state = 42
np.random.seed(rnd_state)
random.seed(rnd_state)


# In[4]:


def display_classification_report(y_true, y_pred):
    display(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T)


# In[5]:


def plot_roc(y_test, preds, ax=None, label='model'):
    if not ax:
        fig, ax = plt.subplots(1, 1)
    fpr, tpr, thresholds = roc_curve(y_test, preds)
    ax.plot(fpr, tpr, lw=2, label=label)
    ax.legend(loc='lower right')
    ax.set_title('ROC curve')
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.annotate(f'AUC: {auc(fpr, tpr):.2}', xy=(.43, .025))
    return ax

def plot_pr(y_test, preds, ax=None, label='model'):
    precision, recall, thresholds = precision_recall_curve(y_test, preds)
    
    if not ax:
        fig, ax = plt.subplots()

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

    return ax


# In[6]:


host="https://code.s3.yandex.net/"


# In[7]:


df = pd.read_csv(host+"datasets/users_behavior.csv")


# In[8]:


sns.countplot(df["is_ultra"])


# In[9]:


df["is_ultra"].value_counts()/len(df)*100


# In[10]:


df.head()


# In[10]:


df.describe()


# In[11]:


df.info()


# In[12]:


target = "is_ultra"
features = ["calls", "minutes", "messages", "mb_used"]


# In[13]:


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


# In[14]:


describe_full(df, target)


# In[15]:


sns.pairplot(df, hue=target, size=3)


# ### Step 2.
# <br>
# <li>Split the source data into a training set, a validation set, and a test set.</li>

# In[13]:


X, y = df[features], df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,                                                    random_state=rnd_state, stratify = y)

X_train, X_val, y_train, y_val     = train_test_split(X_train, y_train, test_size=0.25, random_state=rnd_state) 


# Let's check the validity of our split:

# In[14]:


X_train.shape, X_test.shape


# In[15]:


set(X_train.index) & set(X_test.index)


# In[16]:


set(X_val.index) & set(X_test.index)


# In[17]:


print(f"train = {100*len(X_train)/len(df):.4f}%")
print(f"val = {100*len(X_val)/len(df):.4f}%")
print(f"test = {100*len(X_test)/len(df):.4f}%")


# In[18]:


X_train, X_val, X_test, y_train, y_val, y_test =             X_train.values, X_val.values, X_test.values, y_train.values, y_val.values, y_test.values


# Let's see most important features

# In[22]:


rfc = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = rfc.predict(X_test)
df_feature_importances = pd.DataFrame(((zip(features, rfc.feature_importances_)))).rename(columns={0:"feature",1:"coeff"}).sort_values(by="coeff", ascending = False )
sns.barplot(data=df_feature_importances, x=df_feature_importances["coeff"], y=df_feature_importances["feature"])


# features sorted by importance

# In[23]:


features = list(df_feature_importances["feature"])
features


# ### Step 3, 4, 5
# <br>
# Investigate the quality of different models by changing hyperparameters. Briefly describe the findings of the study.
# 
# 
# <br>
# Check the quality of the model using the test set.
# 
# <br>
# Additional task: sanity check the model. This data is more complex than what you’re used to working with, so it's not an easy task. We'll take a closer look at it later.

# I did 3-5 tasks alltogether, and also used already implemented scikit classes of cross validation 

# ### Baselines:

# #### Dummy Classifier and Gaussian Naive Bayes

# In[24]:


dummy_model = DummyClassifier(strategy='stratified').fit(X_train, y_train)
dummy_preds = dummy_model.predict(X_test)
print(f"mean accuracy on test set: {dummy_model.score(X_test, y_test):.2f}")


# In[25]:


nb_pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('nb', GaussianNB())
]).fit(X_train, y_train)
nb_preds = nb_pipeline.predict(X_test)
print(f"porb of each classes: {nb_pipeline.named_steps['nb'].class_prior_}")
print(f"mean accuracy on test set: {nb_pipeline.named_steps['nb'].score(X_test, y_test):.2f}")


# As we see random guessing baseline around 60% 

# ####  LogisticRegression

# In[26]:


lr = LogisticRegression().fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
display_classification_report(y_test, y_pred)


# In[27]:


lr=LogisticRegression().fit(X_train, y_train)
y_pred = cross_val_predict(lr, X_test, y_test, cv=5)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
display_classification_report(y_test, y_pred)


# #### hyper parameters tuning

# In[28]:


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


# #### DecisionTreeClassifier

# In[29]:


dt = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = dt.predict(X_test)
print(dt.score(X_train, y_train))
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
display_classification_report(y_test, y_pred)


# as we see by default DecisionTreeClassifier just memorized whole dataset and our model became overfitting

# we could search hyper parameters by hand

# In[44]:


train_scores = []
val_scores = []
steps = []
params_array = []

params = {
"criterion":["gini", "entropy"],
"max_depth":[2,4,8,16, 32],
"min_samples_split":[2,4,8, 16],
"min_samples_leaf":[2,4,6,8]}

for step, p in enumerate(ParameterGrid(params), start=1):
    dtc = DecisionTreeClassifier(**p).fit(X_train, y_train)
    train_scores.append(dtc.score(X_train, y_train))
    val_scores.append(dtc.score(X_val, y_val))
    steps.append(step)
    params_array.append(p)
    
train_scores = np.array(train_scores)
val_scores = np.array(val_scores)   
plt.plot(steps, train_scores, color='blue', alpha=0.3, linestyle='dashed')
plt.plot(steps, val_scores, color='red', alpha=0.3, linestyle='dashed')
plt.legend(loc='lower center')
plt.ylim(0, 1)
plt.xlabel('step')
plt.ylabel('score')
scores_df = pd.DataFrame({"diff":train_scores-val_scores, "val_score":val_scores})
print(scores_df.sort_values(['diff', 'val_score'], ascending=[True,False]).iloc[0])
print(np.max(val_scores))
print(params_array[np.argmax(val_scores)]) 


# We could find optimum points, where scores are best, and differences are between train scores and validation scores, and find best parameters(best for this is plotting learning curve probably)

# #### hyper parameters tuning with cross validation

# In[30]:


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


# In[31]:


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


# #### CatBoost

# In[32]:


cb = CatBoostClassifier(verbose=0, random_state=rnd_state).fit(X_train, y_train)
y_pred = cb.predict(X_test)
print(cb.score(X_train, y_train))
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
display_classification_report(y_test, y_pred)

_, axs = plt.subplots(1, 2,figsize=(10,5))
axs = axs.ravel()
plot_pr(y_test, y_pred, ax=axs[0], label="CatBoostClassifier")
plot_roc(y_test, y_pred, ax=axs[1], label="CatBoostClassifier")


# #### XGBoost

# In[33]:


xgb_clf = XGBClassifier().fit(X_train, y_train)
y_train_preds = xgb_clf.predict(X_train)
y_test_preds = xgb_clf.predict(X_test)
# Print F1 scores and Accuracy
print("Training F1 Micro Average: ", f1_score(y_train, y_train_preds, average = "micro"))
print("Test F1 Micro Average: ", f1_score(y_test, y_test_preds, average = "micro"))
print("Test Accuracy: ", accuracy_score(y_test, y_test_preds))


# In[34]:


xgb_clf = XGBClassifier(tree_method = "exact", predictor = "cpu_predictor")

# Create parameter grid
parameters = {"learning_rate": [0.1, 0.01],
               "gamma" : [0.01, 0.3, 0.5, 1, 2],
               "max_depth": [7, 10, 16],
               "colsample_bytree": [0.3, 0.8],
               "subsample": [0.2, 0.5],
               "reg_alpha": [0, 0.5, 1],
               "reg_lambda": [1, 1.5, 2, 3, 4.5],
              "min_child_weight": [1, 3, 5, 7],
               "n_estimators": [100, 500]}

xgb_rscv = RandomizedSearchCV(xgb_clf, param_distributions = parameters, scoring = make_scorer(accuracy_score),
                             cv = 10, verbose = 0)

model_xgboost = xgb_rscv.fit(X, y)


# In[35]:


clf = XGBClassifier(**model_xgboost.best_params_).fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(clf.score(X_train, y_train))
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
display_classification_report(y_test, y_pred)

_, axs = plt.subplots(1, 2,figsize=(10,5))
axs = axs.ravel()
plot_pr(y_test, y_pred, ax=axs[0], label="XGBClassifier")
plot_roc(y_test, y_pred, ax=axs[1], label="XGBClassifier")


# By default, CatBoostClassifier showed better results, but after cross-validation and tuning parameters, XGBClassifier showed optimum: 82% for both sets. On the other hand, I didn't tune parameters for CatBoostClassifier and assume that CatBoostClassifier could do the same job at least not worse that XGBClassifier

# As we see all models showed pretty nice results in spite of that fact that we didn't balance data

# In[144]:


ros = RandomOverSampler(random_state = rnd_state)
X = df[features]
y = df[target]
X_resampled, y_resampled = ros.fit_resample(X, y)

new_df = pd.DataFrame(X_resampled, columns = features)
new_df[target] = y_resampled
new_df[target].value_counts()


# In[145]:



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,                                                    random_state=rnd_state, stratify = y)

X_train, X_val, y_train, y_val     = train_test_split(X_train, y_train, test_size=0.25, random_state=rnd_state) 


# In[146]:


cb = CatBoostClassifier(verbose=0, random_state=rnd_state).fit(X_train, y_train)
y_pred = cb.predict(X_test)
print(cb.score(X_train, y_train))
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
display_classification_report(y_test, y_pred)

_, axs = plt.subplots(1, 2,figsize=(10,5))
axs = axs.ravel()
plot_pr(y_test, y_pred, ax=axs[0], label="CatBoostClassifier")
plot_roc(y_test, y_pred, ax=axs[1], label="CatBoostClassifier")


# ## Summary

# as we saw , we have pretty good data, we showed how to obtain around 82% of optimum of accuracy score on both sets using cross validation and hyper parameters technics. Also we saw that attempt of balancing data in this case didn't give us some impressive advantage
