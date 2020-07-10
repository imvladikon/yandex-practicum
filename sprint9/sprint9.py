import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import *
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, LogisticRegressionCV
from sklearn.utils import shuffle
from scipy import stats as st
from statsmodels.api import OLS
import random
random.seed(42)
np.random.seed(42)
random_state = np.random.RandomState(42)
import warnings
warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

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

def display_group_density_plot(df, groupby, on, palette, figsize):
    """
    Displays a density plot by group, given a continuous variable, and a group to split the data by
    :param df: DataFrame to display data from
    :param groupby: Column name by which plots would be grouped (Categorical, maximum 10 categories)
    :param on: Column name of the different density plots
    :param palette: Color palette to use for drawing
    :param figsize: Figure size
    :return: matplotlib.axes._subplots.AxesSubplot object
    """

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

    # Plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')

    for value, color in zip(groups, palette):
        sns.kdeplot(df.loc[df[groupby] == value][on], \
                    shade=True, color=color, label=value)

    ax.set_title(str("Distribution of " + on + " per " + groupby + " group"),\
                 fontsize=30)
    
    ax.set_xlabel(on, fontsize=20)
    return ax 

geo_data_0 = pd.read_csv("https://code.s3.yandex.net/datasets/geo_data_0.csv")
geo_data_1 = pd.read_csv("https://code.s3.yandex.net/datasets/geo_data_1.csv")
geo_data_2 = pd.read_csv("https://code.s3.yandex.net/datasets/geo_data_2.csv")

geo_data_0.isna().sum()

geo_data_1.isna().sum()

geo_data_2.isna().sum()

describe_full(geo_data_0)

describe_full(geo_data_1)

describe_full(geo_data_2)

target = "product"
features = ['f0', 'f1', 'f2']

gd1 = geo_data_0.copy()
gd1["source"] = "geo_data_0"
gd2 = geo_data_1.copy()
gd2["source"] = "geo_data_1"
gd3 = geo_data_2.copy()
gd3["source"] = "geo_data_2"
df = pd.concat([gd1, gd2, gd3])
del gd1,gd2,gd3

mdf = pd.melt(df[[target, 'source']], id_vars=['source'], var_name=[target])
ax = sns.boxplot(x="source", y="value", hue=target, data=mdf)    
plt.show()
plt.clf()

display_group_density_plot(mdf, groupby = "source", on = 'value', \
                                           palette = sns.color_palette('Set2'), 
                           figsize = (10, 5))

def test_lr_model(X_train,X_valid,y_train,y_valid):
  lr = LinearRegression().fit(X_train,y_train)
  y_pred = lr.predict(X_valid)  
  return mean_squared_error(y_valid, y_pred,squared=False), lr.score(X_valid, y_valid), y_pred

X_train0,X_valid0,y_train0,y_valid0 = train_test_split(geo_data_0[features].values,geo_data_0[target].values,test_size=0.25, random_state=42)
rmse, score, y_pred0 = test_lr_model(X_train0,X_valid0,y_train0,y_valid0)
print(f"RMSE: {rmse}")
print(f"R^2: {score}")
print(f"y_pred_mean: {y_pred0.mean()}")

X_train1,X_valid1,y_train1,y_valid1 = train_test_split(geo_data_1[features].values,geo_data_1[target].values,test_size=0.25, random_state=42)
rmse, score, y_pred1 = test_lr_model(X_train1,X_valid1,y_train1,y_valid1)
print(f"RMSE: {rmse}")
print(f"R^2: {score}")
print(f"y_pred_mean: {y_pred1.mean()}")

X_train2,X_valid2,y_train2,y_valid2 = train_test_split(geo_data_2[features].values,geo_data_2[target].values,test_size=0.25, random_state=42)
rmse, score, y_pred2 = test_lr_model(X_train2,X_valid2,y_train2,y_valid2)
print(f"RMSE: {rmse}")
print(f"R^2: {score}")
print(f"y_pred_mean: {y_pred2.mean()}")

def test_model_cv(df):
  """
  cross-validation checker of linear regression model 
  (also we could use LinearRegressionCV)
  """
  shuffle(df)
  X, y = df[features].values, df[target].values
  result = dict(rmse=[], score=[])
  for train_index, valid_index in KFold(n_splits=5).split(X):
    scaler = StandardScaler()
    X_train, X_valid = X[train_index], X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)
    rmse, score, _ = test_lr_model(X_train, X_valid, y_train, y_valid)
    result["rmse"].append(rmse)
    result["score"].append(score)
  return pd.DataFrame(result)

test_model_cv(geo_data_0).describe().T['mean']

test_model_cv(geo_data_1).describe().T['mean']

test_model_cv(geo_data_2).describe().T['mean']

n_points_all = 500
n_points = 200
budget_total = 100_000_000 #for 200
income_per_volume = 4500
thres_risk_max = 2.5/100
budget_per_one = budget_total/n_points
unit_of_volume = 1000 #Barrels

#the bad scenario
(budget_total*n_points_all/n_points)/n_points/income_per_volume

#best
print(f"the volume of reserves sufficient for developing a new well without losses = {budget_total/n_points/income_per_volume:.2f}")

def revenue(target, predicted, count):
    indices = predicted.sort_values(ascending=False).index
    return target[indices][:count].sum() * income_per_volume - budget_total

def revenue(target, predicted, well_count):
    predicted_sorted = predicted.sort_values(ascending=False)
    selected = target[predicted_sorted.index][:well_count]
    return selected.sum() * income_per_volume - budget_total

def revenue_bs(target, predicted):
    values = []
    target = pd.Series(target)
    predicted = pd.Series(predicted)
    for i in range(1000):
        target_sample = target.sample(n=n_points_all, replace=True, random_state=random_state)
        values.append(revenue(target_sample, predicted[target_sample.index], n_points))
    return pd.Series(values)

revenues0 = revenue_bs(y_valid0, y_pred0)
revenues1 = revenue_bs(y_valid1, y_pred1)
revenues2 = revenue_bs(y_valid2, y_pred2)

len(revenues0[revenues0<0])/len(revenues0)

len(revenues1[revenues1<0])/len(revenues1)

len(revenues2[revenues2<0])/len(revenues2)

sns.distplot(revenues0)
sns.distplot(revenues1)
sns.distplot(revenues2)
plt.axvline(0, c="r", label="")
plt.legend()

confidence_interval = st.t.interval(0.95, len(revenues0)-1, revenues0.mean(), revenues0.sem())
loss_risk = len(revenues0[revenues0 < 0]) / len(revenues0)
print(f"""average profit of first region is {revenues0.mean():.2f}, 95% confidence interval is  ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f}) \
and risk of losses {loss_risk:.2%}""")

confidence_interval = st.t.interval(0.95, len(revenues1)-1, revenues1.mean(), revenues1.sem())
loss_risk = len(revenues1[revenues1 < 0]) / len(revenues1)
print(f"""average profit of second region is {revenues1.mean():.2f}, 95% confidence interval is  ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f}) \
and risk of losses {loss_risk:.2%}""")

confidence_interval = st.t.interval(0.95, len(revenues2)-1, revenues2.mean(), revenues2.sem())
loss_risk = len(revenues2[revenues2 < 0]) / len(revenues2)
print(f"""average profit of third region is {revenues2.mean():.2f}, 95% confidence interval is  ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f}) \
and risk of losses {loss_risk:.2%}""")
