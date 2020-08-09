#!/usr/bin/env python
# coding: utf-8

# # Review
# Hi, my name is Dmitry and I will be reviewing your project.
# 
# You can find my comments in colored markdown cells:
# 
# <div class="alert alert-success">
#     If everything is done succesfully.
# </div>
# 
# <div class="alert alert-info">
#     If I have some (optional) suggestions, or questions to think about, or general comments.
# </div>
# 
# <div class="alert alert-danger">
#     If a section requires some corrections. Work can't be accepted with red comments.
# </div>
# 
# First of all, thank you for turning in the project! While there is room for improvement, on the whole your project is looking pretty good. There are a couple of problems that need to be fixed, but don't worry! You've got this!

# >Hi! Great, thanks!

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import *
from sklearn.ensemble import *
from sklearn.tree import *
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.utils import shuffle
from sklearn.dummy import DummyRegressor

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


# In[2]:


pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)


# <div class="markdown markdown_size_normal markdown_type_theory"><h1>Project description</h1><div class="paragraph">The data is stored in three files:</div><ul><li><code class="code-inline code-inline_theme_light">gold_recovery_train.csv</code> — training dataset d<a href="https://code.s3.yandex.net/datasets/gold_recovery_train.csv">ownload</a></li><li><code class="code-inline code-inline_theme_light">gold_recovery_test.csv</code> — test dataset <a href="https://code.s3.yandex.net/datasets/gold_recovery_test.csv">download</a></li><li><code class="code-inline code-inline_theme_light">gold_recovery_full.csv</code> — source dataset <a href="https://code.s3.yandex.net/datasets/gold_recovery_full.csv">download</a></li></ul><div class="paragraph">Data is indexed with the date and time of acquisition (<code class="code-inline code-inline_theme_light">date</code> feature). Parameters that are next to each other in terms of time are often similar.</div><div class="paragraph">Some parameters are not available because they were measured and/or calculated much later. That's why, some of the features that are present in the training set may be absent from the test set. The test set also doesn't contain targets.</div><div class="paragraph">The source dataset contains the training and test sets with all the features.</div><div class="paragraph">You have the raw data that was only downloaded from the warehouse. Before building the model, check the correctness of the data. For that, use our instructions.</div>

# <h2>Data description</h2><div class="paragraph"><strong>Technological process</strong></div><ul><li><em>Rougher feed</em> — raw material</li><li><em>Rougher additions</em> (or <em>reagent additions</em>) — flotation reagents: <em>Xanthate, Sulphate, Depressant</em>
#   <ul><li><em>Xanthate</em> — promoter or flotation activator;</li><li><em>Sulphate</em> — sodium sulphide for this particular process;</li><li><em>Depressant</em> — sodium silicate.</li></ul></li><li><em>Rougher process</em>  — flotation</li><li><em>Rougher tails</em> — product residues</li><li><em>Float banks</em> — flotation unit</li><li><em>Cleaner process</em> — purification</li><li><em>Rougher Au</em> — rougher gold concentrate</li><li><em>Final Au</em> — final gold concentrate</li></ul><div class="paragraph"><strong>Parameters of stages</strong></div><ul><li><em>air amount — volume of air</em></li><li><em>fluid levels</em></li><li><em>feed size</em> — feed particle size</li><li><em>feed rate</em></li></ul><h2>Feature naming</h2><div class="paragraph">Here's how you name the features:</div><div class="paragraph"><code class="code-inline code-inline_theme_light">[stage].[parameter_type].[parameter_name]</code></div><div class="paragraph">Example: <code class="code-inline code-inline_theme_light">rougher.input.feed_ag</code></div><div class="paragraph">Possible values for <code class="code-inline code-inline_theme_light">[stage]</code>:</div><ul><li><em>rougher —</em> flotation</li><li><em>primary_cleaner</em> — primary purification</li><li><em>secondary_cleaner</em> — secondary purification</li><li><em>final</em> — final characteristics</li></ul><div class="paragraph">Possible values for <code class="code-inline code-inline_theme_light">[parameter_type]</code>:</div><ul><li><em>input</em> — raw material parameters</li><li><em>output</em> — product parameters</li><li><em>state</em> — parameters characterizing the current state of the stage</li><li><em>calculation —</em> calculation characteristics</li></ul>

# In[3]:


def display_group_density_plot(df, groupby, on, palette = None, figsize = None, title="", ax=None): 
    """
    Displays a density plot by group, given a continuous variable, and a group to split the data by
    :param df: DataFrame to display data from
    :param groupby: Column name by which plots would be grouped (Categorical, maximum 10 categories)
    :param on: Column name of the different density plots
    :param palette: Color palette to use for drawing
    :param figsize: Figure size
    :return: matplotlib.axes._subplots.AxesSubplot object
    """
    if palette is None:
      palette = sns.color_palette('Set2')
    if figsize is None:
      figsize = (10, 5)
    if not isinstance(df, pd.core.frame.DataFrame):
        raise ValueError('df must be a pandas DataFrame')

    if not groupby:
        raise ValueError('groupby parameter must be provided')

    elif not groupby in df.keys():
        raise ValueError(groupby + ' column does not exist in the given DataFrame')

    if not on:
        raise ValueError('on parameter must be provided')

    elif not on in df.keys():
        raise ValueError(on + ' column does not exist in the given DataFrame')

    if len(set(df[groupby])) > 10:
        groups = df[groupby].value_counts().index[:10]

    else:
        groups = set(df[groupby])

    # Get relevant palette
    if palette:
        palette = palette[:len(groups)]
    else:
        palette = sns.color_palette()[:len(groups)]

    if ax is None:
      fig = plt.figure(figsize=figsize)
      ax = fig.add_subplot(111)
    
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    for value, color in zip(groups, palette):
        sns.kdeplot(df.loc[df[groupby] == value][on],                     shade=True, color=color, label=value, ax=ax)
    if not title:
      title = str("Distribution of " + on + " per " + groupby + " group")
    
    ax.set_title(title,fontsize=16)
    ax.set_xlabel(on, fontsize=16)
    return ax 


# In[4]:


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


# 1. Prepare the data

# 1.1. Open the files and look into the data.
# Path to files:
# * /datasets/gold_recovery_train.csv
# * /datasets/gold_recovery_test.csv
# * /datasets/gold_recovery_full.csv

# In[5]:


df_train = pd.read_csv("https://code.s3.yandex.net/datasets/gold_recovery_train.csv")
df_test = pd.read_csv("https://code.s3.yandex.net/datasets/gold_recovery_test.csv")
df_full = pd.read_csv("https://code.s3.yandex.net/datasets/gold_recovery_full.csv")


# In[6]:


df_full.info()


# 1.2. Check that recovery is calculated correctly. Using the training set, calculate recovery for the rougher.output.recovery feature. Find the MAE between your calculations and the feature values. Provide findings.

# In[7]:


list(filter(lambda s:"rougher" in s,df_train.columns))


# In[8]:


rougher_output_recovery_calc = 100 * (df_train['rougher.output.concentrate_au'] * (df_train['rougher.input.feed_au'] - df_train['rougher.output.tail_au'])) / (df_train['rougher.input.feed_au'] * (df_train['rougher.output.concentrate_au'] - df_train['rougher.output.tail_au']))
df_output_rougher = pd.DataFrame({"output_recovery":df_train["rougher.output.recovery"],"calc":rougher_output_recovery_calc}).dropna()
MAE = mean_absolute_error(df_output_rougher["output_recovery"],df_output_rougher["calc"])
print(f"MAE={MAE}")
del df_output_rougher


# as we could see, result is correct

# <div class="alert alert-success">
#     Good job checking the recovery data validity!
# </div>

# 1.3. Analyze the features not available in the test set. What are these parameters? What is their type?

# In[9]:


df_types = df_full.dtypes.reset_index()
df_types.columns = ["name", "type"]
all_columns = pd.Series(list(set(df_full.columns).union(set(df_train.columns).union(set(df_test.columns)))))
df_temp = pd.DataFrame({"name": all_columns}).merge(df_types).sort_values(by="name")
df_temp["full"] = np.vectorize(lambda x: int(x in df_full.columns))(df_temp["name"].values)
df_temp["train"] = np.vectorize(lambda x: int(x in df_train.columns))(df_temp["name"].values)
df_temp["test"] = np.vectorize(lambda x: int(x in df_test.columns))(df_temp["name"].values)
df_temp[df_temp["test"]==0]


# In[10]:


describe_full(df_train)


# In[11]:


describe_full(df_test)


# <div class="alert alert-success">
#     Great! Now we know which features are unavailable in the test set
# </div>

# 1.4. Perform data preprocessing.

# In[12]:


for df in [df_full, df_train, df_test]:
  if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)


# In[13]:


print(df_train['final.output.recovery'].isna().sum())
print(df_train['rougher.output.recovery'].isna().sum())


# In[14]:


df_test.merge(df_full[['final.output.recovery', 'rougher.output.recovery']], how='left', left_index=True, right_index=True).head(2)


# In[15]:


df_test = df_test.merge(df_full[['final.output.recovery', 'rougher.output.recovery']], how='left', left_index=True, right_index=True)


# Let's fill missing data using nearest values strategy - ffill, because our gathering of our data relates to datetime (it's like in a famous joke, what's the best strategy for the weather prediction of the next day? weather of the current day)

# In[16]:


df_train = df_train[~df_train['final.output.recovery'].isna()]
df_train = df_train[~df_train['rougher.output.recovery'].isna()]
df_train = df_train.fillna(method='ffill')


# In[17]:


df_test = df_test[~df_test['final.output.recovery'].isna()]
df_test = df_test[~df_test['rougher.output.recovery'].isna()]
df_test = df_test.fillna(method='ffill')


# In[18]:


df_train.info()


# <div class="alert alert-success">
#     Well done! You prepared the data and dealt with missing values
# </div>

# 2. Analyze the data

# 2.1. Take note of how the concentrations of metals (Au, Ag, Pb) change depending on the purification stage.

# 2.2. Compare the feed particle size distributions in the training set and in the test set. If the distributions vary significantly, the model evaluation will be incorrect.

# 2.3. Consider the total concentrations of all substances at different stages: raw feed, rougher concentrate, and final concentrate. Do you notice any abnormal values in the total distribution? If you do, is it worth removing such values from both samples? Describe the findings and eliminate anomalies.

# In[57]:


metals = ['au', 'ag', 'pb']
columns = ['rougher.output.tail', 'primary_cleaner.output.tail','final.output.tail',           'rougher.output.concentrate', 'primary_cleaner.output.concentrate', 'final.output.concentrate']

_, axs = plt.subplots(2,3, figsize=(21, 9))
axs = axs.flatten()

for column, ax in zip(columns, axs):
    cols = [f"{t}_{m}" for t,m in zip([column]*3,metals)]
    temp_df = pd.melt(df_train[cols])
    temp_df["variable"] = temp_df["variable"].str.replace(column+"_","")
    display_group_density_plot(temp_df, groupby="variable", on="value", title=f"distribution of {column} per metals", ax=ax)
    ax.legend()

plt.tight_layout()
plt.show()


# we could see that there are some abnormal values of the concentration, especially in rougher and primary_cleaner stages. assume, that when first readouts had come about a concentration of the pb, ag, and on this stage we got some outliers around zero.   

# In[61]:


metals = ['au', 'ag', 'pb']
columns = ['rougher.output.tail', 'primary_cleaner.output.tail','final.output.tail',           'rougher.output.concentrate', 'primary_cleaner.output.concentrate', 'final.output.concentrate']

_, axs = plt.subplots(2,3, figsize=(21, 9))
axs = axs.flatten()

for column, ax in zip(columns, axs):
    cols = [f"{t}_{m}" for t,m in zip([column]*3,metals)]
    temp_df = pd.melt(df_train[cols])
    temp_df["variable"] = temp_df["variable"].str.replace(column+"_","")
    sns.boxplot(x="variable", y="value", hue="variable", data=temp_df, ax=ax)
    ax.legend()

plt.tight_layout()
plt.show()


# In[21]:


metals = ['au', 'ag', 'pb']
column = 'rougher.input.feed'
cols = [f"{t}_{m}" for t,m in zip([column]*3,metals)]
temp_df = pd.melt(df_train[cols])
temp_df["variable"] = temp_df["variable"].str.replace(column+"_","")
ax = display_group_density_plot(temp_df, groupby="variable", on="value", title=f"distribution of {column} per metals")
temp_df = pd.melt(df_test[cols])
temp_df["variable"] = temp_df["variable"].str.replace(column+"_","")
display_group_density_plot(temp_df, groupby="variable", on="value", title=f"distribution of {column} per metals", ax=ax)
plt.tight_layout()
plt.show()


# In[29]:


list(filter(lambda s: "feed_size" in s, df_train.columns))


# In[56]:


sns.distplot(df_train["primary_cleaner.input.feed_size"],bins=15, hist_kws=dict(alpha=0.7), label="train")
g = sns.distplot(df_test["primary_cleaner.input.feed_size"],bins=15, hist_kws=dict(alpha=0.3), label="test")
g.set(xlim=(0, 20))
g.legend()


# In[52]:


sns.distplot(df_train["rougher.input.feed_size"],bins=100, hist_kws=dict(alpha=0.7), label="train")
g = sns.distplot(df_test["rougher.input.feed_size"],bins=100, hist_kws=dict(alpha=0.3), label="test")
g.set(xlim=(0, 150))
g.legend()


# In[27]:


df_train.describe()['rougher.input.feed_size']


# In[22]:


df_test.describe()['rougher.input.feed_size']


# #### Summary:

# According to density plots the feed particle size distributions in the training set and in the test set is differenet, but I'd say so, it's not vary significantly, also we could see that mean values are really close to each other

# as we could see, concentration of the gold was increased, but plumbum and silver after final stage were decreased or quite the same
# (gold's concentration is increasing in a linear way. it's seems there is some chemistry property of the plumbum,  it's not changing so much in the final stage and silver's concetration is decreasing to a final stage)

# <div class="alert alert-success">
#     Great, you've studied the concentration changes, so we know that the process is doing what it's supposed to.
# </div>

# <div class="alert alert-danger">
#     1. You seem to compare "input.feed" distributions instead of "input.feed_size" distributions. <br>
#     2. Although you studied total concentrations, it seems that you missed the abnormal values near 0.
# </div>

# > * 1. it's true, it seems that I confused;) fixed it
#   * I've noticed it, but didn't know how to explain, added explanation. also I think, that we need to leave this values for model, because it's better also to teach model on this abnormal values(on plots we could see that abnormal values near 0 are not just simple outliers)

# 3. Build the model

# 3.1. Write a function to calculate the final sMAPE value.

# 3.2. Train different models. Evaluate them using cross-validation. Pick the best model and test it using the test sample. Provide findings.
# Use these formulas for evaluation metrics:
# 
# $sMAPE= \frac{1}{N} \sum_{i=1}^{N} \frac {|y_i-\hat{y_i}|}{(|y_i|+|\hat{y_i}|)/2} \times 100\% $
# 
# $Final\hspace{0.2cm}sMAPE = 25\% \times sMAPE(rougher) + 75\% \times sMAPE(final)$

# In[23]:


target = ['rougher.output.recovery', 'final.output.recovery']
features = list(set(df_train.columns).intersection(set(df_test.columns)).difference(set(target)))


# In[24]:


def smape(y_true, y_pred):
    frac = np.divide(np.abs(y_true - y_pred), (np.abs(y_true)+np.abs(y_pred))/2)
    return np.average(frac, axis=0)


# In[25]:


def smape_final(y_true,y_pred):
    smape_out_rougher = smape(y_true[target.index('rougher.output.recovery')], y_pred[target.index('rougher.output.recovery')])
    smape_out_final = smape(y_true[target.index('final.output.recovery')], y_pred[target.index('final.output.recovery')])
    return 0.25*smape_out_rougher + 0.75*smape_out_final


# <div class="alert alert-success">
#     Metric calculation is absolutely correct!
# </div>

# In[36]:


print(df_train.isna().sum().sum())
print(df_test.isna().sum().sum())


# In[27]:


df_train = df_train.dropna()
df_test = df_test.dropna()


# In[41]:


smape_score = make_scorer(smape_final)


# In[29]:


X_train, X_test = df_train[features].values, df_test[features].values
y_train, y_test = df_train[target].values, df_test[target].values


# In[42]:


lr = LinearRegression().fit(X_train, y_train)
scores_lr = cross_val_score(lr, X_train, y_train, cv=5, scoring=smape_score)
print("mean smape:", scores_lr.mean())
scores_lr


# In[43]:


params = {'min_samples_split': range(2, 10, 2), 'max_depth': range(4,8,2)}
g_cv = GridSearchCV(DecisionTreeRegressor(random_state=random_state),param_grid=params,scoring=smape_score, cv=5, refit=True)
g_cv.fit(X_train, y_train)
best_params = g_cv.best_params_


# In[44]:


dtr = DecisionTreeRegressor(**best_params).fit(X_train, y_train)
scores_dtr = cross_val_score(dtr, X_train, y_train, cv=5, scoring=smape_score)
print("mean smape:", scores_dtr.mean())
scores_dtr


# In[ ]:


params = {'min_samples_split': range(2, 6, 2)}
rf_cv = GridSearchCV(RandomForestRegressor(random_state=random_state),param_grid=params,scoring=smape_score, cv=5, refit=True)
rf_cv.fit(X_train, y_train)
best_params = rf_cv.best_params_


# In[47]:


rfr = RandomForestRegressor(**best_params).fit(X_train, y_train)
scores_rfr = cross_val_score(rfr, X_train, y_train, cv=5, scoring=smape_score)
print("mean smape:", scores_rfr.mean())
scores_rfr


# let's check with our baseline - DummyRegressor

# In[48]:


dm = DummyRegressor(strategy='mean').fit(X_train, y_train)
y_pred = dm.predict(X_test)
print('smape:', smape_final(y_test, y_pred))


# #### Summary:
# 
# we compared several models with Dummy Regressor (sMAPE=0.054):
# 
# 1) Linear regression: $sMAPE_{mean}\approx$0.0614
# 
# 2) Decision tree regressor: $sMAPE_{mean}\approx$0.0614
# 
# 3) Random forest regressor: $sMAPE_{mean}\approx$0.1087
# 
# Linear regression and Decision tree regressor have the best result according to the sMAPE

# <div class="alert alert-info">
#     The model training section is great: you try a few models, correctly use the train and test sets, use cross-validation, do some hyper-parameter tuning, and have a baseline. The only thing that is strange to me is that you say that linear regression and decision tree regressor have the best results, but don't say that they are both outperformed by a very dumb constant model predicting the mean of train set targets.
# </div>

# >that's true;) 

# Project evaluation
# * We’ve put together the evaluation criteria for the project. Read this carefully before moving on to the case.
# * Here’s what the reviewers will look at when reviewing your project:
# * Have you prepared and analyzed the data properly?
# * What models have you developed?
# * How did you check the model‘s quality?
# * Have you followed all the steps of the instructions?
# * Did you keep to the project structure and explain the steps performed?
# * What are your findings?
# * Have you kept the code neat and avoided code duplication?
