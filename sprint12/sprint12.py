#!/usr/bin/env python
# coding: utf-8

# ## Project description

# Rusty Bargain used car sales service is developing an app to attract new customers. In that app, you can quickly find out the market value of your car. You have access to historical data: technical specifications, trim versions, and prices. You need to build the model to determine the value. 
# 
# Rusty Bargain is interested in:
# 
# - the quality of the prediction;
# - the speed of the prediction;
# - the time required for training

# <div class="markdown markdown_size_normal markdown_type_theory"><h3>Project instructions</h3><ol start="1"><li>Download and look at the data.</li><li>Train different models with various hyperparameters (You should make at least two different models, but more is better. Remember, various implementations of gradient boosting don't count as different models.) The main point of this step is to compare gradient boosting methods with random forest, decision tree, and linear regression.</li><li>Analyze the speed and quality of the models.</li></ol><div class="paragraph">Notes:</div><ul><li>Use the <em>RMSE</em> metric to evaluate the models.</li><li>Linear regression is not very good for hyperparameter tuning, but it is perfect for doing a sanity check of other methods. If gradient boosting performs worse than linear regression, something definitely went wrong.</li><li>On your own, work with the LightGBM library and use its tools to build gradient boosting models.</li><li>Ideally, your project should include linear regression for a sanity check, a tree-based algorithm with hyperparameter tuning (preferably, random forrest), LightGBM with hyperparameter tuning (try a couple of sets), and CatBoost and XGBoost with hyperparameter tuning (optional).</li><li>Take note of the encoding of categorical features for simple algorithms. LightGBM and CatBoost have their implementation, but XGBoost requires OHE.</li><li>You can use a special command to find the cell code runtime in Jupyter Notebook. Find that command.</li><li>Since the training of a gradient boosting model can take a long time, change only a few model parameters.</li></ul></div>

# In[1]:


get_ipython().system('pip install -qq catboost')


# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import *
from sklearn.ensemble import *
from sklearn.tree import *
from sklearn.linear_model import *
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.dummy import DummyRegressor
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import random
random_state=42
random.seed(random_state)
np.random.seed(random_state)

import warnings
warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
import timeit
from functools import lru_cache


# # 1. Data preparation

# <div class="markdown markdown_size_normal markdown_type_theory"><h3>Data description</h3><div class="paragraph">The dataset is stored in file <code class="code-inline code-inline_theme_light">/datasets/car_data.csv</code>. <a href="https://code.s3.yandex.net/datasets/car_data.csv" target="_blank">download dataset</a>.</div><div class="paragraph"><strong>Features</strong></div><ul><li><em>DateCrawled</em> — date profile was downloaded from the database</li><li><em>VehicleType</em> — vehicle body type</li><li><em>RegistrationYear</em> — vehicle registration year</li><li><em>Gearbox</em> — gearbox type</li><li><em>Power</em> — power (hp)</li><li><em>Model</em> — vehicle model</li><li>Mileage — mileage (measured in km due to dataset's regional specifics)</li><li><em>RegistrationMonth</em> — vehicle registration month</li><li><em>FuelType</em> — fuel type</li><li><em>Brand</em> — vehicle brand</li><li><em>NotRepaired</em> — vehicle repaired or not</li><li><em>DateCreated</em> — date of profile creation</li><li><em>NumberOfPictures</em> — number of vehicle pictures</li><li><em>PostalCode</em> —  postal code of profile owner (user)</li><li><em>LastSeen</em> — date of the last activity of the user</li></ul><div class="paragraph"><strong>Target</strong></div><div class="paragraph"><em>Price</em> — price (Euro)</div></div>

# #### Helper functions:

# In[3]:


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
    data_describe['outliers %']=[outlier(df_numeric[col])/len(df) * 100 for col in df_numeric.columns]
    data_describe['kurtosis']=df_numeric.kurtosis()
    return data_describe

def display_classification_report(y_true, y_pred):
    display(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T)


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

def show_feature_importances(df, features, target):
  X, y = df[features].values,df[target].values
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
  rfc = DecisionTreeRegressor().fit(X_train, y_train)
  y_pred = rfc.predict(X_test)
  df_feature_importances = pd.DataFrame(((zip(features, rfc.feature_importances_)))).rename(columns={0:"feature",1:"coeff"}).sort_values(by="coeff", ascending = False )
  sns.barplot(data=df_feature_importances, x=df_feature_importances["coeff"], y=df_feature_importances["feature"])
  return df_feature_importances


# In[4]:


get_ipython().system('wget https://code.s3.yandex.net/datasets/car_data.csv -O ./project/car_data.csv')


# In[5]:


url = "https://code.s3.yandex.net/datasets/car_data.csv"


# In[6]:


df =  pd.read_csv(url)


# In[7]:


target = "Price"
features = list(set(df.columns)-set(target))


# In[8]:


df.head()


# In[9]:


df.info()


# #### type conversion

# In[10]:


df["DateCrawled"] = pd.to_datetime(df["DateCrawled"])
df["DateCreated"] = pd.to_datetime(df["DateCreated"])
df["LastSeen"] = pd.to_datetime(df["LastSeen"])


# #### imputing

# good question how to impute NotRepaired. NaN imho is deafualt field, and it means no one changed it. and it's hard to understand what is real meaning of it. let's think that it's 'yes' (it's more probable meaning of default value)

# In[11]:


df["NotRepaired"] = df["NotRepaired"].fillna('yes')
df["NotRepaired"] = (df["NotRepaired"] == 'yes').astype('int')


# In[12]:


df["Gearbox"] = (df["Gearbox"] == "auto").astype("int")


# let's impute others categorical features by model depends on others values and without target (for avoiding leakage in the reaulting model)

# In[13]:


df.isna().sum()


# In[14]:


def impute_value(in_df, features, target):
  encoders=dict()
  df = in_df.copy()
  for col in df[features].select_dtypes('object').columns:
    df.loc[df[col].isna(), col] = "None"
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  
  for col in df[features].select_dtypes('datetime64').columns:
    df[f"{col}_hour"] = df[col].dt.hour
    df[f"{col}_month"] = df[col].dt.month
    df[f"{col}_day"] = df[col].dt.day  
    del df[col]
  features = list(set(df.columns)-set([target]))
  train_df = df[~df[target].isna()]
  test_df = df[df[target].isna()]
  let = LabelEncoder()
  y_train = let.fit_transform(train_df[target])
  y_train = train_df[target].values
  X_train, X_test = train_df[features].values, test_df[features].values
  if len(X_test)==0:
    return in_df
  model = DecisionTreeClassifier().fit(X_train, y_train)
  y_pred = model.predict(X_test)
  df.loc[df[target].isna(), target] = y_pred
  in_df[target] = df[target]
  return in_df


# In[15]:


for col in ["FuelType", "VehicleType", "Model"]:
  df = impute_value(df, features=list(set(df.columns)-set([col])-set([target])), target=col)


# In[16]:


df.isna().sum()


# In[17]:


describe_full(df, target_name=target)


# # 2. Model training

# let's use CV technics and 10% for a final test set. 
# we gonna see 3 models - CatBoost, XGBoost, and LGBMBoost

# In[18]:


def get_data(df, transform_data=True, apply_encoding=False):
  in_df = df.copy()
  target = "Price"
  features = list(set(in_df.columns)-set([target]))
  if transform_data:
    for col in in_df[features].select_dtypes('datetime64').columns:
      in_df[f"{col}_hour"] = in_df[col].dt.hour
      in_df[f"{col}_month"] = in_df[col].dt.month
      in_df[f"{col}_day"] = in_df[col].dt.day  
      del in_df[col]
  features = list(set(in_df.columns)-set([target]))
  encoders = dict()
  if apply_encoding:
    for col in in_df[features].select_dtypes('object').columns:
      lbl = LabelEncoder().fit(in_df[col].values)
      in_df[col] = lbl.transform(in_df[col].values)
      encoders[col] = lbl
  features = list(set(in_df.columns)-set([target]))
  cat_features = list(in_df[features].select_dtypes('object').columns)
  return in_df[features].values, in_df[target].values, features, target, encoders, cat_features


# In[19]:


rmse_func = lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False)
rmsle  = make_scorer(rmse_func, greater_is_better=False)


# In[20]:


def eval_model(model, X_train, X_test, y_train, y_test):
  model.fit(X_train, y_train, eval_metric='rmse', verbose = False, eval_set = [(X_test, y_test)])
  y_pred = model.predict(X_test)
  print("RMSE", rmse_func(y_test, y_pred))


# In[21]:


log_metrics = {"models": ["catboost", "xgboost", "LGBM"], "rmse_init": [0.0]*3, "rmse_cv": [0.0]*3}
m_idx = {"catboost":0, "xgboost":1, "LGBM":2}


# CatBoost

# CatBoost by default could work with categorical data (that's why it's CatBoost;)) for others model, we gonna use just simple LabelEncoder for simplicity (in real case I believe need to think better, because some categorical data for sure is sensitive to order, in terms of the price for example)

# In[22]:


X, y, features, target, _, cat_features = get_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)
train_ds = Pool(data=X_train, label=y_train, cat_features=cat_features, feature_names=features)
test_ds = Pool(data=X_test, label=y_test, cat_features=cat_features, feature_names=features)
full_ds = Pool(data=X, label=y, cat_features=cat_features, feature_names=features)


# In[23]:


model = CatBoostRegressor(iterations=20, task_type="GPU", devices='0:1', random_seed=random_state, loss_function='RMSE', has_time=True)
model.fit(train_ds, verbose = 0)
y_pred = model.predict(test_ds)
print("RMSE", rmse_func(y_test, y_pred))
print("time {}".format(timeit.timeit()))


# In[24]:


log_metrics["rmse_init"][m_idx["catboost"]] = rmse_func(y_test, y_pred)


# CatBoost, GridSearch

# In[25]:


param_grid = {
        'learning_rate': [0.03, 0.1],
        'depth': [6, 10],
        'l2_leaf_reg': [3, 5, 7, 9],
        'has_time': [True]        
}

model = CatBoostRegressor(iterations=20, loss_function='RMSE', task_type="GPU", devices='0:1', random_seed=random_state)
grid_search_result = model.grid_search(param_grid, 
                                       full_ds,
                                       verbose=0,
                                       partition_random_seed=random_state,
                                       search_by_train_test_split=True,
                                       train_size=0.9,
                                       plot=False)
print("time {}".format(timeit.timeit()))


# In[26]:


cv_data = pd.DataFrame(grid_search_result["cv_results"])
best_value = cv_data['test-RMSE-mean'].min()
best_iter = cv_data['test-RMSE-mean'].values.argmin()

print('Best validation RMSE score : {:.4f}±{:.4f} on step {}'.format(
    best_value,
    cv_data['test-RMSE-std'][best_iter],
    best_iter)
)
print("time {}".format(timeit.timeit()))


# In[27]:


model = CatBoostRegressor(iterations=20, loss_function='RMSE', task_type="GPU", devices='0:1', random_seed=random_state, **grid_search_result["params"])
model.fit(train_ds, verbose = 1, eval_set = [(X_test, y_test)], use_best_model=True)
y_pred = model.predict(test_ds)
print("RMSE", rmse_func(y_test, y_pred))
print("time {}".format(timeit.timeit()))


# In[28]:


log_metrics["rmse_cv"][m_idx["catboost"]] = rmse_func(y_test, y_pred)


# XGBoost

# In[29]:


X, y, features, target, encoders, _ = get_data(df, apply_encoding=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)


# In[30]:


model = XGBRegressor(tree_method='gpu_hist', gpu_id=0, random_state=random_state, objective='reg:squarederror')
eval_model(model, X_train, X_test, y_train, y_test)
print("time {}".format(timeit.timeit()))


# In[31]:


y_pred = model.predict(X_test)
log_metrics["rmse_init"][m_idx["xgboost"]] = rmse_func(y_test, y_pred)


# XGBoost, CV

# In[32]:


param_grid = {
        'learning_rate': [0.03, 0.1],
        'max_depth': [4, 6, 10],
        'objective':['reg:squarederror']
        
}
model = XGBRegressor(tree_method='gpu_hist', gpu_id=0, random_state=random_state)

grid = GridSearchCV(model,
                        param_grid,
                        cv = 5,
                        n_jobs = 5,
                        verbose=False,
                        scoring=rmsle)

grid.fit(X, y)

print(grid.best_params_)
print("time {}".format(timeit.timeit()))


# In[33]:


model = XGBRegressor(tree_method='gpu_hist', gpu_id=0, random_state=random_state, **grid.best_params_)
eval_model(model, X_train, X_test, y_train, y_test)
print("time {}".format(timeit.timeit()))


# In[34]:


y_pred = model.predict(X_test)
log_metrics["rmse_cv"][m_idx["xgboost"]] = rmse_func(y_test, y_pred)
print("time {}".format(timeit.timeit()))


# LGBMRegressor

# don't want recompile it on GPU, run it on CPU

# In[35]:


X, y, features, target, encoders, _ = get_data(df, apply_encoding=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)


# In[36]:


model = LGBMRegressor(objective="RMSE", random_state=random_state, verbose=1)
eval_model(model, X_train, X_test, y_train, y_test)
print("time {}".format(timeit.timeit()))


# In[37]:


y_pred = model.predict(X_test)
log_metrics["rmse_init"][m_idx["LGBM"]] = rmse_func(y_test, y_pred)
print("time {}".format(timeit.timeit()))


# In[38]:


param_grid = {
        'learning_rate': [0.03, 0.1],
        'max_depth': [4, 6, 10]
}

model = LGBMRegressor(objective="RMSE", random_state=random_state, verbose=0)

grid = GridSearchCV(model,
                        param_grid,
                        cv = 5,
                        n_jobs = 5,
                        verbose=False)

grid.fit(X, y)

print(grid.best_params_)
print("time {}".format(timeit.timeit()))


# In[39]:


model = LGBMRegressor(objective="RMSE", random_state=random_state, verbose=0, **grid.best_params_)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
log_metrics["rmse_cv"][m_idx["LGBM"]] = rmse_func(y_test, y_pred)
print("{}".format(timeit.timeit()))


# # 3. Model analysis

# Let's check our results, first of all, I didn't show measure of time executions, because I used GPU on the couple models, and on the last one on the CPU. for sure time would be different (anyway time results are pretty the same)

# In[40]:


pd.DataFrame(log_metrics)


# second thing is 
# Best(lowest) RMSE has LGBM for init settings and XGBoost in the cross-validation
# I believe that good result CatBoost could show, it could work with categorical features from the box and could manage with time-related data. But I didn't spend a lot of time for manage settings for finetuned it. And suspicious that CV showed worser result compare with init
# 
# Best result is on the XGBoost, and LGBM has weighted result in both cases
# 
# 

# ## Checklist

# Type 'x' to check. Then press Shift+Enter.

# - [x]  Jupyter Notebook is open
# - [ ]  Code is error free
# - [ ]  The cells with the code have been arranged in order of execution
# - [ ]  The data has been downloaded and prepared
# - [ ]  The models have been trained
# - [ ]  The analysis of speed and quality of the models has been performed

# In[40]:




