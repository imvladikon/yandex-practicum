#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-info">
# <h2> Comments </h2>
# </div>
# 
# Hi Vladimir,
# 
# I have checked (just a part of it) you work and left comments in such cells. Cells are of two types:
# 
# <div class="alert alert-block alert-danger">
# <p> <strong> A red colored cell </strong> indicates that you need to improve or adjust part of the project above. </p>
# </div>
# <div class="alert alert-block alert-info">
# <p> <strong> A blue colored cell </strong> indicates that no improvements are needed in the cells above. May include some suggestions and recommendations.</p>
# </div>
# 
# Hope it all will be clear to you :)
# 
# There is a error in cell 36 and that is why I was not able to check following parts. Please fix this problem.
# 
# ------------

# <div class="alert alert-block alert-info">
# <h2> Comments 1 </h2>
# </div>
# 
# Thank you for fixing the problem :)
# 
# Unfortunately, now there is a problem in cell 38: there is no such variable called `pos`. Please fix it :)
# 
# ------------

# sorry, was inattentive ;)

# <div class="alert alert-block alert-info">
# <h2> Comments 2 </h2>
# </div>
# 
# Hi :)
# 
# Thank you for fixing all the problems :) Comments on this version are in such cells.
# 
# Project is accepted - good jod:)
# 
# I would suggest that you comment more on what you do and what results you get in further projects.
# 
# *Good luck!*
# 
# ------------

# # Research on apartment sales ads
# 
# You will have the data from a real estate agency. It is an archive of sales ads for realty in St. Petersburg, Russia, and the surrounding areas collected over the past few years. You’ll need to learn how to determine the market value of real estate properties. Your task is to define the parameters. This will make it possible to build an automated system that is capable of detecting anomalies and fraudulent activity.
# 
# There are two different types of data available for every apartment for sale. The first type is a user’s input. The second type is received automatically based upon the map data. For example, the distance from the downtown area, airport, the nearest park or body of water. 

# ### Step 1. Open the data file and study the general information. 

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy import stats
import numpy as np
import math
from sklearn.preprocessing import FunctionTransformer
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# Business model - we have data collected by 2 methods. Our target feature is a price, as real estate agency
# we want understand what kind of factors could impact on our price, 
# and what features have positive impact and what features have negative impact.
# Also we want to have some model that could detect anomalies, - missing data from user input, or fraudulent activity
# It means that potentially we have skewness in data, missing data and outliers.

# In[2]:


filename='https://code.s3.yandex.net/datasets/real_estate_data_eng.csv'


# In[3]:


filename_ru='https://code.s3.yandex.net/datasets/real_estate_data.csv'


# In[4]:


estate_ru_df = pd.read_csv(filename_ru, sep = '\t')


# In[5]:


estate_df = pd.read_csv(filename, sep = '\t')
estate_df.head()


# In[6]:


estate_df.sample(5)


# In[7]:


estate_df.describe(include=["float"]).T


# In[8]:


estate_df.columns


# In[9]:


estate_df.info()


# In[10]:


df_nulls=pd.concat([estate_df.dtypes, estate_df.isna().sum(), estate_df.isna().sum()/len(estate_df)], axis=1)
df_nulls.columns = ["type","count","missing_ratio"]
df_nulls=df_nulls[df_nulls["count"]>0]
df_nulls.sort_values(by="missing_ratio", ascending=False)


# In[11]:


estate_df.isnull().sum().sum()


# In[12]:


df_nulls.shape


# In[13]:


estate_df.describe()


# In[14]:


sns.distplot(estate_df['last_price'])


# In[15]:


sns.boxplot(estate_df['last_price'])


# as we see we have outliers (or our scale should be non linear, e.g. loglinear)

# In[16]:


estate_df['last_price'].describe()


# In[17]:


sns.boxplot(estate_df[estate_df['last_price']<6.800000e+06]['last_price'])


# In[18]:


f"Skewness: {estate_df['last_price'].skew()}"
f"Kurtosis: {estate_df['last_price'].kurt()}"


# Q.E.D. last price data is skewed

# In[19]:


def describe_full(df, target_name):
    data_describe = df.describe().T
    df_numeric=df._get_numeric_data()
    dtype_df=df_numeric.dtypes
    data_describe['dtypes']=dtype_df
    data_null = df_numeric.isnull().sum()/len(df) * 100
    data_describe['Missing %']=data_null
    Cardinality=df_numeric.apply(pd.Series.nunique)
    data_describe['Cardinality']=Cardinality
    df_skew=df_numeric.skew(axis = 0, skipna = True) 
    data_describe['Skew']=df_skew
    return data_describe
describe_full(estate_df, "last_price")


# In[20]:


estate_df["total_images"].hist()


# In[21]:


cols = ["total_images",
"total_area",
"rooms",
"ceiling_height",
"floors_total",
"living_area",
"floor",
"kitchen_area",
"balcony",
"airports_nearest",
"cityCenters_nearest",
"parks_around3000",
"parks_nearest",
"ponds_around3000",
"ponds_nearest",
"days_exposition"]
for col in cols:
    estate_df[[col, "last_price"]].corr()


# In[22]:


sns.pairplot(data=estate_df,
                  y_vars=['last_price'],
                  x_vars=["days_exposition" , "total_images"])
sns.pairplot(data=estate_df,
                  y_vars=['last_price'],
                  x_vars=["total_area","rooms","ceiling_height","floors_total","living_area","floor","kitchen_area","balcony"])
sns.pairplot(data=estate_df,
                  y_vars=['last_price'],
                  x_vars=["airports_nearest","cityCenters_nearest","parks_around3000","parks_nearest","ponds_around3000","ponds_nearest"])


# In[23]:


estate_df[['last_price']].plot(kind = 'kde')


# In[24]:


transformer = FunctionTransformer(np.log1p, validate=True)
pd.DataFrame(transformer.transform(estate_df[['last_price']])).plot(kind = 'kde')


# In[25]:


estate_df['llast_price'] = transformer.transform(estate_df[['last_price']])


# In[26]:


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
        sns.kdeplot(df.loc[df[groupby] == value][on],                     shade=True, color=color, label=value)

    ax.set_title(str("Distribution of " + on + " per " + groupby + " group"),                 fontsize=30)
    
    ax.set_xlabel(on, fontsize=20)
    return ax 


# In[27]:


display_group_density_plot(estate_df, groupby = "total_area", on = 'llast_price',                                            palette = sns.color_palette('Set2'), 
                           figsize = (10, 5))


# In[28]:


sns.pairplot(data=estate_df,
                  y_vars=['llast_price'],
                  x_vars=["days_exposition" , "total_images"])
sns.pairplot(data=estate_df,
                  y_vars=['llast_price'],
                  x_vars=["total_area","rooms","ceiling_height","floors_total","living_area","floor","kitchen_area","balcony"])
sns.pairplot(data=estate_df,
                  y_vars=['llast_price'],
                  x_vars=["airports_nearest","cityCenters_nearest","parks_around3000","parks_nearest","ponds_around3000","ponds_nearest"])


# ### Conclusion

# There are a lot of missing data. Also we could see correlation between some features, but first of all we need get rid off missing data (or fill it)
# It seems that is_apartment is redundant column, it has no apparrent correlation with target column (last_price)
# Also target column has skewness that could bother for building good model. We need to fit columns, scale them. Because as
# we saw above we have deal with imbalanced data
# 

# <ul>
#     <li><em>airports_nearest</em>
#         <ul>
#             <li>the distance to the nearest airport in meters (m.).</li>
#             <li>a lot of missing (23.4%)</li>
#             <li>numerical</li>
#         </ul>
#     </li>
#     <li><em>balcony</em>
#         <ul>
#             <li>the number of balconies.</li>
#             <li>a lot of missing</li>
#             <li>numerical</li>
#         </ul>
#     </li>
#     <li><em>ceiling_height</em>
#         <ul>
#             <li>the ceiling height in meters (m.).</li>
#             <li>38.8% missing</li>
#             <li>numerical</li>
#         </ul>
#     </li>
#     <li><em>cityCenters_nearest</em>
#         <ul>
#             <li>the distance to the Saint-Petersburg center in meters (m.).</li>
#             <li>23.3% missing</li>
#             <li>numerical</li>
#         </ul>
#     </li>
#     <li><em>days_exposition</em>
#         <ul>
#             <li>how many days the ad was displayed (from publication to removal).</li>
#             <li>13.4% missing</li>
#             <li>numerical</li>
#         </ul>
#     </li>
#     <li>
#         <em>first_day_exposition -</em> the publication date.
#         <ul>
#             <li>datetime</li>
#         </ul>
#     </li>
#     <li><em>floor</em>
#         <ul>
#             <li>the apartment floor number.</li>
#         </ul>
#     </li>
#     <li><em>floors_total</em>
#         <ul>
#             <li>the total number of floors in the building.</li>
#             <li>missing 0.4%</li>
#         </ul>
#     </li>
#     <li><em>is_apartment</em>
#         <ul>
#             <li>legacy column which doesn't convey any important information (Boolean type). See the note below.</li>
#             <li>Boolean</li>
#             <li>88.3%!!! missing</li>
#             <li>candidate for deleting</li>
#         </ul>
#     </li>
#     <li><em>kitchen_area</em>
#         <ul>
#             <li>the kitchen area in square meters (sq.m.).</li>
#             <li>9.6% missing</li>
#             <li>numerical</li>
#         </ul>
#     </li>
#     <li><em>last_price</em>
#         <ul>
#             <li>the price at the time when the ad was removed.</li>
#         </ul>
#     </li>
#     <li><em>living_area</em>
#         <ul>
#             <li>the living area in square meters (sq.m.).</li>
#             <li>8% missing</li>
#             <li>high correlation</li>
#         </ul>
#     </li>
#     <li><em>locality_name</em>
#         <ul>
#             <li>the locality name.</li>
#             <li>0.2% missing</li>
#             <li>categorical, ordinal</li>
#         </ul>
#     </li>
#     <li><em>open_plan</em>
#         <ul>
#             <li>an open plan design (Boolean type).</li>
#             <li>Boolean</li>
#         </ul>
#     </li>
#     <li><em>parks_around3000</em>
#         <ul>
#             <li>the number of parks in a 3 km. radius.</li>
#             <li>23.3% missing</li>
#             <li>categorical</li>
#             <li>have gueesing for simplicity turn this column to Boolean</li>
#         </ul>
#     </li>
#     <li><em>parks_nearest</em>
#         <ul>
#             <li>the distance to the nearest park in meters (m.).</li>
#             <li>65.9% missing</li>
#             <li>numerical</li>
#         </ul>
#     </li>
#     <li><em>ponds_around3000</em>
#         <ul>
#             <li>the number of bodies of water in a 3 km. radius.</li>
#             <li>23.3% missing like in parks_around3000</li>
#             <li>categorical</li>
#         </ul>
#     </li>
#     <li><em>ponds_nearest</em>
#         <ul>
#             <li>the distance to the nearest body of water (m.).</li>
#             <li>61.6% misisng</li>
#             <li>numerical</li>
#         </ul>
#     </li>
#     <li><em>rooms</em>
#         <ul>
#             <li>the number of bedrooms.</li>
#         </ul>
#     </li>
#     <li><em>studio</em>
#         <ul>
#             <li>whether it's a studio or not (Boolean type).</li>
#             <li>Boolean</li>
#         </ul>
#     </li>
#     <li><em>total_area</em>
#         <ul>
#             <li>the total area in square meters (sq.m.).</li>
#         </ul>
#     </li>
#     <li><em>total_images</em>
#         <ul>
#             <li>the number of photos of the apartment in the ad.</li>
#         </ul>
#     </li>
# </ul>

# <div class="alert alert-block alert-info">
# <h2> Comments </h2>
# </div>
# 
# Okay :)
# 
# Do not forget to name graphs and their axis :)
# 
# ------------

# ### Step 2. Data preprocessing

# <ul><li>Determine and study the missing values:
#   <ul><li>A practical replacement can be presumed for some missing values. For example, if the user doesn’t enter the number of balconies, then there probably aren’t any. The correct course of action here is to replace these missing values with 0. There’s no suitable replacement value for other data types. In this case, leave these values blank. A missing value is also a key indicator that mustn’t be hidden.</li><li>Fill in the missing values where appropriate. Explain why you’ve chosen to fill the missing values in these particular columns and how you selected the values.</li><li>Describe the factors that may have led up to the missing values.</li></ul></li><li>Convert the data to the required types:
#   <ul><li>Indicate the columns where the data types have to be changed and explain why.</li></ul></li></ul>

# #### is_apartment

# Note: the column 'is_apartment' has True values for properties that belong to the Russian real estate entity confusingly called "apartments" - a space that is non-residential in a legal sense and is not considered a part of the housing stock, even though it is often used for temporary rental housing. This column has nothing to do with whether the property in question is an apartment or not - all records in this dataset are apartments.

# In[29]:


sns.scatterplot(x="is_apartment", y="llast_price", data=estate_df)


# A lot of missing data. Maybe there is some influence on last_price in valid data of is_apartment, but it's not our case
# Let's drop it

# In[30]:


estate_df = estate_df.drop("is_apartment", axis=1)


# In[31]:


df_nulls=pd.concat([estate_df.dtypes, estate_df.isna().sum(), estate_df.isna().sum()/len(estate_df)], axis=1)
df_nulls.columns = ["type","count","missing_ratio"]
df_nulls=df_nulls[df_nulls["count"]>0]
df_nulls.sort_values(by="missing_ratio", ascending=False)


# <div class="alert alert-block alert-info">
# <h2> Comments </h2>
# </div>
# 
# Okay :)
# 
# ------------

# #### locality_name

# In[32]:


estate_df[estate_df["locality_name"].isna()]


# let's infer locality from geo metric

# In[33]:


df_locality_city_cntr = estate_df[~estate_df["locality_name"].isna()][["locality_name", "cityCenters_nearest"]].drop_duplicates().dropna()[:].reset_index()
df_locality_city_cntr


# In[34]:


df_locality_city_cntr.groupby("locality_name").mean()


# In[35]:


len(estate_df["locality_name"].unique())
len(estate_df["locality_name"].str.replace("village","").str.strip().unique())
set(estate_df["locality_name"].unique()) - set(estate_df["locality_name"].str.replace("village","").str.strip().unique())


# In[36]:


get_ipython().system('pip install aiohttp')


# In[37]:


### GEO UTILS ###

from math import asin, sqrt, pow, sin, cos, pi
from numpy import rad2deg, deg2rad

RADIUS_METERS = 6378137

import requests
import aiohttp
import asyncio

spb_loc = (30.3609, 59.9311)

async def get_pos(locality):
    url = "https://geocode-maps.yandex.ru/1.x/?apikey=9dc2f628-77ef-4202-ad14-913e947114f4&format=json&geocode={}&ll=30.3609,59.9311&results=1".format(
        locality)
    async with aiohttp.ClientSession() as session:
        json = await fetch(session, url)
        lon, lat = None, None
        try:
            response = json["response"]["GeoObjectCollection"]["featureMember"][0]["GeoObject"]["Point"]["pos"]
            response = response.split() 
            lon, lat = float(response[0]), float(response[1])
        except:
            pass
        return lon, lat


async def fetch(session, url):
    async with session.get(url) as response:
        return await response.json()
    
def calc_dist_km(t1, t2):
    """
    :param t1: (LONGITUDE, LATITUDE)
    :param t2:  (LONGITUDE, LATITUDE)
    :return: calculate distance(meters) between points (WGS84)
    """
    if t1 is None or t2 is None:
        return 0
    lng1, lat1 = t1
    lng1, lat1 = deg2rad(lng1), deg2rad(lat1)
    lng2, lat2 = t2
    lng2, lat2 = deg2rad(lng2), deg2rad(lat2)
    return 2 * asin(
        sqrt(pow(sin((lat1 - lat2) / 2), 2) + cos(lat1) * cos(lat2) * pow(sin((lng1 - lng2) / 2), 2))) * RADIUS_METERS/1000

def dist_line_km(line):
    return sum(calc_dist_km(prev, curr) for prev, curr in prev_curr(line) if not prev is None)


# In[38]:


pos_locality = {}
for locality in estate_ru_df["locality_name"].unique():
    pos = await get_pos(locality)
    pos_locality[locality] = pos


# In[39]:


dist_locality = {}
for locality_name, pos in pos_locality.items():
    dist_locality[locality_name] = calc_dist_km(spb_loc, pos)


# In[40]:


estate_ru_df["pos"] = estate_ru_df["locality_name"].apply(lambda l: pos_locality[l]) 


# In[41]:


estate_ru_df["dist"] = estate_ru_df["locality_name"].apply(lambda l: dist_locality[l]) 


# In[42]:


estate_df["pos"] = estate_ru_df["pos"]
estate_df["dist"] = estate_ru_df["dist"]*1000


# In[43]:


#probably geocoding couldn't work in a good way with case when locality name is Spb itself (it's large city)
estate_df[estate_df["locality_name"]!="Saint Peterburg"][["cityCenters_nearest", "dist", "locality_name"]]


# In[44]:


estate_df.loc[(estate_df["cityCenters_nearest"].isna()) & (estate_df["locality_name"]!="Saint Peterburg"), "cityCenters_nearest"] = estate_df["dist"]


# In[45]:


estate_df[estate_df["cityCenters_nearest"].isna()]


# <div class="alert alert-block alert-danger">
# <h2> Comments </h2>
# </div>
# 
# Here is a probem 😓 Can you fix it?
# 
# ------------

# <div class="alert alert-block alert-danger">
# <h2> Comments </h2>
# </div>
# 
# Fixed it 
# 
# ------------

# In[46]:


estate_df["locality_name"] = estate_df["locality_name"].str.replace("village","").str.strip()


# In[47]:


estate_median_locality_df = estate_df.pivot_table(index='locality_name', values=['airports_nearest', 'cityCenters_nearest'],aggfunc='median')
estate_median_locality_df


# In[48]:


def fill_dist_metric(row, col):
    locality = row['locality_name']
    if locality in estate_median_locality_df.index:
        return estate_median_locality_df.loc[locality][col]
    return row[col]
for column in ['airports_nearest', 'cityCenters_nearest']:
    estate_df.loc[estate_df[column].isnull(), column] = estate_df.apply(fill_dist_metric, axis=1, args=(column,))


# In[49]:


estate_df['airports_nearest'] = estate_df['airports_nearest'].fillna(-1)
estate_df['cityCenters_nearest'] = estate_df['cityCenters_nearest'].fillna(-1)


# In[50]:


estate_df[estate_df['cityCenters_nearest'].isnull()]["locality_name"]


# <div class="alert alert-block alert-info">
# <h2> Comments 2 </h2>
# </div>
# 
# Great 🔥🔥🔥
# 
# ------------

# #### first_day_exposition

# In[51]:


estate_df["first_day_exposition"].dtype


# In[52]:


estate_df["first_day_exposition"] = pd.to_datetime(estate_df["first_day_exposition"])
estate_df["first_day_exposition"].dtype


# In[53]:


estate_df["first_day_exposition"].sample(5)


# #### cityCenters_nearest,parks_around3000, ponds_around3000

# Those columns have approx. the same missing values but cityCenters_nearest. Let's check diff

# airports_nearest
# cityCenters_nearest
# parks_around3000
# parks_nearest
# ponds_around3000
# ponds_nearest

# In[54]:


estate_df[(estate_df["cityCenters_nearest"].isna()) & (~estate_df["parks_around3000"].isna())].T


# In[55]:


df_city_centers_pushkin = estate_df[(estate_df["locality_name"]=="Pushkin") | (estate_df["airports_nearest"]==15527)]["cityCenters_nearest"]
df_city_centers_pushkin.describe()
sns.boxplot(df_city_centers_pushkin)


# It's pretty around. We could assign mean value in one case where cityCenters_nearest is na and parks_around3000 is vice verca

# In[56]:


estate_df[(estate_df["cityCenters_nearest"].isna()) & (~estate_df["parks_around3000"].isna())]["cityCenters_nearest"]=df_city_centers_pushkin.mean()


# In[57]:


estate_df[(estate_df["cityCenters_nearest"].isna()) & (estate_df["parks_around3000"].isna())]


# #### balcony

# Likelihood that missing value of balcony it's when users didn't write anything if balcony's value was zero

# In[58]:


estate_df["balcony"].fillna(0, inplace=True)


# In[59]:


estate_df["balcony"].isna().sum()


# #### days_exposition

# In[60]:


estate_df[estate_df["days_exposition"].isna()]


# In[61]:


estate_df[estate_df["days_exposition"]==0]
estate_df[estate_df["days_exposition"]==1]
estate_df[estate_df["days_exposition"]==2]


# In[62]:


estate_df["days_exposition"].hist(bins=50)


# In[63]:


estate_df[~estate_df["days_exposition"].isna()][["days_exposition"]].groupby("days_exposition").agg({"days_exposition":"count"}).head(10)


# It seems that missed days are between 1 and 2. 

# <div class="alert alert-block alert-info">
# <h2> Comments 2 </h2>
# </div>
# 
# Okay :)
# 
# ------------

# #### kitchen area

# In[64]:


estate_df['kitchen_area'] = estate_df['kitchen_area'].fillna(estate_df['kitchen_area'].median())
estate_df['kitchen_area'].isna().sum()


# <div class="alert alert-block alert-info">
# <h2> Comments 2 </h2>
# </div>
# 
# Filling in living and kitchen area with median area can result in sum of living and kitchen area being greater than the total area of the observation, which is not realistic at all. To make this approach more realistic you should at least fill nan values with respect to the number of rooms in the ad. Another way to deal with these nulls is to study living to total area ratio and kitchen to total area ration and fill nans based on these ratios. These ratios are quite similar for all types of ads.
# 
# ------------

# #### ceiling_height

# In[65]:


estate_df['ceiling_height'].describe()


# Assume that ceil height around 3.0m, and value 100 is incorrect

# In[66]:


estate_df.loc[estate_df['ceiling_height']==estate_df['ceiling_height'].max(), 'ceiling_height'] = np.nan


# In[67]:


estate_df.loc[estate_df['ceiling_height']>4]['ceiling_height'].hist(bins=30)


# In[68]:


estate_df.loc[estate_df['ceiling_height']>10]


# strange value, let's fill na by median, because mean is perceptive to outliers

# In[69]:


estate_df['ceiling_height'] = estate_df['ceiling_height'].fillna(estate_df['ceiling_height'].median())
estate_df['ceiling_height'].isna().sum()


# <div class="alert alert-block alert-info">
# <h2> Comments 2 </h2>
# </div>
# 
# Okay :)
# 
# ------------

# #### living_area

# Let's fill it by median, but assuming that it could not be less than kithen area(according common sense)

# In[70]:


living_area_median = estate_df['living_area'].median()
estate_df['living_area'] = estate_df.apply(lambda row: max(living_area_median, row['kitchen_area']) if np.isnan(row["living_area"]) else row["living_area"], axis=1)
estate_df['living_area'].isna().sum()


# #### floors_total

# Depends of building, we don't have some specific features that could point on total floors (Sure, we could not use 
# for that target column). try to see distribution, calculate depends on it, mean, locality and also floors_total>=floor 

# In[71]:


estate_df["floors_total"].hist(bins=60)


# In[72]:


estate_df["floors_total"].describe()
sns.boxplot(estate_df["floors_total"])


# as we see, we have strange data. (outliers) unlikely that there is 60 floors, I googled, in Saint Petersburg area max floor is 42

# In[73]:


estate_df[estate_df["floors_total"]>30][["floors_total", "locality_name", "cityCenters_nearest", "airports_nearest"]]


# In[74]:


estate_df[estate_df["floors_total"]>36][["floors_total", "locality_name", "cityCenters_nearest", "airports_nearest"]]


# I checked, building with 37 floors is real (it seems that it's  Alexander Nevsky (residential complex)). 

# In[75]:


estate_df[(estate_df["locality_name"]=="Kronshtadt")]["floors_total"].describe()


# It's seems that it's just typo. 60 instead 6

# In[76]:


estate_df.loc[estate_df["floors_total"]==60, "floors_total"] = 6


# In[77]:


mean_floors_estate_df = estate_df.groupby('locality_name').agg({'floors_total': 'mean', 'floor':'max'}).reset_index()
mean_floors_estate_df["floors_total"] = mean_floors_estate_df["floors_total"].astype(np.int)  
replacer_map = {row["locality_name"]:max(row["floors_total"],row["floor"]) for i, row in mean_floors_estate_df.iterrows()}
estate_df['floors_total'] = estate_df['floors_total'].fillna(estate_df["locality_name"].map(replacer_map))


# In[78]:


estate_df.loc[estate_df["floors_total"].isnull(), "floors_total"] = estate_df[estate_df["floors_total"].isnull()]["floor"]


# In[79]:


estate_df["floors_total"] = estate_df["floors_total"].astype(np.int)


# <div class="alert alert-block alert-info">
# <h2> Comments 2 </h2>
# </div>
# 
# Okay :)
# 
# ------------

# ### Step 3. Make calculations and add them to the table

# <ul><li>the price per square meter</li><li>the day of the week, month, and year that the ad was published</li><li>which floor the apartment is on (first, last, or other)</li><li>the ratio between the living space and the total area, as well as between the kitchen space and the total area.</li></ul>

# #### the price per square meter

# In[80]:


def price_per_square_meter(row):
    total_area = row['total_area']
    return row['last_price']/total_area if total_area!=0 else 0

estate_df['price_per_square_meter'] = estate_df.apply(price_per_square_meter, axis=1)


# #### the day of the week, month, and year that the ad was published

# In[81]:


estate_df["weekday"] = estate_df["first_day_exposition"].dt.weekday
estate_df["month"] = estate_df["first_day_exposition"].dt.month
estate_df["year"] = estate_df["first_day_exposition"].dt.year
estate_df["day"] = estate_df["first_day_exposition"].dt.day


# #### which floor the apartment is on (first, last, or other)

# In[82]:


def floor_type(row):
    floor = row['floor']
    floors_total = row['floors_total']
    if floor == 1 :
        return 'first'
    elif floor == floors_total:
        return 'last'
    return 'other'

estate_df['floor_type'] = estate_df.apply(floor_type, axis=1)


# In[83]:


estate_df["first_floor"] = (estate_df["floor"] == 1).astype(np.int)


# In[84]:


estate_df["last_floor"] = (estate_df["floor"] == estate_df["floors_total"]).astype(np.int)


# In[85]:


estate_df["other_floor"] = (estate_df["floor"] != 1) & (estate_df["floor"] != estate_df["floors_total"]).astype(np.int)


# #### the ratio between the living space and the total area, as well as between the kitchen space and the total area.

# In[86]:


def living_area_ratio(row):
    total_area = row['total_area']
    return row['living_area']/total_area if total_area!=0 else 0

estate_df['living_area_ratio'] = estate_df.apply(living_area_ratio, axis=1)


# In[87]:


def kitchen_area_ratio(row):
    total_area = row['total_area']
    return row['kitchen_area']/total_area if total_area!=0 else 0

estate_df['kitchen_area_ratio'] = estate_df.apply(kitchen_area_ratio, axis=1)


# In[88]:


sns.pairplot(data=estate_df,
                  x_vars=['last_price'],
                  y_vars=["kitchen_area_ratio" , "living_area_ratio"])


# In[89]:


estate_df[["price_per_square_meter", "kitchen_area_ratio" , "living_area_ratio"]].describe()


# <div class="alert alert-block alert-info">
# <h2> Comments 2 </h2>
# </div>
# 
# Good :)
# 
# ------------

# ### Step 4. Conduct exploratory data analysis and follow the instructions below:

# <ul><li>Carefully investigate the following parameters: square area, price, number of rooms, and ceiling height. Plot a histogram for each parameter.</li><li>Examine the time it's taken to sell the apartment and plot a histogram. Calculate the mean and median and explain the average time it usually takes to complete a sale. When can a sale be considered to have happened rather quickly or taken an extra long time?</li><li>Remove rare and outlying values and describe the patterns you've discovered.</li><li>Which factors have had the biggest influence on an apartment’s price? Examine whether the value depends on the total square area, number of rooms, floor (top or bottom), or the proximity to the downtown area. Also study the correlation to the publication date: day of the week, month, and year.</li><li>Select the 10 localities with the largest number of ads then calculate the average price per square meter in these localities. Determine which ones have the highest and lowest housing prices. You can find this data by name in the ’<em>locality_name’</em> column.</li><li>Thoroughly look at apartment offers: Each apartment has information about the distance to the city center. Select apartments in Saint Petersburg (<em>‘locality_name’</em>). Your task is to pinpoint which area is considered to be downtown. In order to do that, create a column with the distance to the city center in km and round to the nearest whole number. Next, calculate the average price for each kilometer and plot a graph to display how prices are affected by the distance to the city center. Find a place on the graph where it shifts significantly. That's the downtown border.</li><li>Select all the apartments in the downtown and examine correlations between the following parameters: total area, price, number of rooms, ceiling height. Also identify the factors that affect an apartment’s price: number of rooms, floor, distance to the downtown area, and ad publication date. Draw your conclusions. Are they different from the overall deductions about the entire city?</li></ul>

# In[90]:


estate_df["ceiling_height"].plot(kind='hist', bins=20,range=(2,4), title="the ceil of the apartment on the number of sales")
plt.show()


# In[91]:


estate_df["total_area"].plot(kind='hist', bins=100,range=(0,400), title="the area of the apartment on the number of sales")
plt.show()


# In[92]:


estate_df["rooms"].plot(kind='hist', bins=10,range=(0,7), title="the number of the rooms of the apartment on the number of sales")
plt.show()


# In[93]:


estate_df["last_price"].plot(kind='hist', bins=120,range=(200_000, 15_000_000), title="the price of the apartment on the number of sales")
plt.show()


# In[94]:


estate_df['days_exposition'].describe()


# In[95]:


sns.boxplot(data=estate_df, x="days_exposition")


# In[96]:


f"mean:{estate_df['days_exposition'].mean()}"
f"median: {estate_df['days_exposition'].median()}"


# In[97]:


estate_df['days_exposition'].hist()


# In[98]:


display_group_density_plot(estate_df, groupby = "total_area", on = 'last_price',                                            palette = sns.color_palette('Set2'), 
                           figsize = (10, 5))


# The total square area, number of rooms, floor (top or bottom), or the proximity to the downtown area. Also study the correlation to the publication date: day of the week, month, and year.

# In[100]:


display_group_density_plot(estate_df, groupby = "cityCenters_nearest", on = 'last_price',                                            palette = sns.color_palette('Set2'), 
                           figsize = (4, 5))


# In[101]:


estate_df.columns


# Conclusion: avg. apartment sale time 95 days.

# In[102]:


cols = [
 'living_area_ratio', 'kitchen_area_ratio',    
"total_area",
"rooms",
"ceiling_height",
"floors_total",
"living_area",
"floor",
"kitchen_area",
"balcony",
"airports_nearest",
"cityCenters_nearest",
"parks_around3000",
"parks_nearest",
"ponds_around3000",
"ponds_nearest",
"days_exposition"]
for col in cols:
    estate_df[[col, "last_price"]].corr()


# As we see there is high correlation between square metric and price

# In[103]:


spb_estate_df = estate_df.loc[estate_df['locality_name'] == 'Saint Peterburg'][:]


# In[104]:


sns.scatterplot(y="last_price", x='cityCenters_nearest', data=spb_estate_df.pivot_table(index='cityCenters_nearest', values='last_price').reset_index())


# In[106]:


sns.pairplot(data=spb_estate_df,
                  y_vars=['last_price'],
                  x_vars=["days_exposition" , "total_images"])
sns.pairplot(data=spb_estate_df,
                  y_vars=['last_price'],
                  x_vars=["total_area","rooms","ceiling_height","floors_total","living_area","floor","kitchen_area","balcony"])
sns.pairplot(data=spb_estate_df,
                  y_vars=['last_price'],
                  x_vars=["airports_nearest","cityCenters_nearest","parks_around3000","parks_nearest","ponds_around3000","ponds_nearest"])


# In[105]:


display_group_density_plot(spb_estate_df, groupby = "total_area", on = 'last_price',                                            palette = sns.color_palette('Set2'), 
                           figsize = (10, 5))


# In[107]:


spb_estate_df.describe().T[["50%","mean"]]


# <div class="alert alert-block alert-info">
# <h2> Comments 2 </h2>
# </div>
# 
# Okay :)
# 
# You need to give comments to each graph you've plotted, table you've builed or value you've calculated.
# 
# ------------

# ### Step 5. Overall conclusion

# Overall conclusion. 
# Fisrt of all as expected we have some problems with missing data (since we have two different sources of collecting data)
# The distance metrics also were missed. I tried filled it. (Idea was using geocoding, but api wasn't responded)
# There were found metrics that have impact on our target feature. Built graphs of the dependence of the price of the apartment on its parameters. The parameters that directly affect the cost of the apartment are defined:
# Also were created new features. Also we found that the most of the apartments are for the city center, were the number of rooms decreases.
# 
# 

# <div class="alert alert-block alert-info">
# <h2> Comments 2 </h2>
# </div>
# 
# Okay :)
# 
# What are those features that affect cost of the apartment?
# 
# ------------

# ### Project completion checklist
# 
# Mark the completed tasks with 'x'. Then press Shift+Enter.

# - [x]  file opened
# - [ ]  files explored (first rows printed, info() method)
# - [ ]  missing values determined
# - [ ]  missing values filled in
# - [ ]  clarification of the discovered missing values provided
# - [ ]  data types converted
# - [ ]  explanation of which columns had the data types changed and why
# - [ ]  calculated and added to the table: the price per square meter
# - [ ]  calculated and added to the table: the day of the week, month, and year that the ad was published
# - [ ]  calculated and added to the table: which floor the apartment is on (first, last, or other)
# - [ ]  calculated and added to the table: the ratio between the living space and the total area, as well as between the kitchen space and the total area
# - [ ]  the following parameters investigated: square area, price, number of rooms, and ceiling height
# - [ ]  histograms for each parameter created
# - [ ]  task completed: "Examine the time it's taken to sell the apartment and create a histogram. Calculate the mean and median and explain the average time it usually takes to complete a sale. When can a sale be considered extra quick or taken an extra slow?"
# - [ ]  task completed: "Remove rare and outlying values and describe the specific details you've discovered."
# - [ ]  task completed: "Which factors have had the biggest influence on an apartment’s value? Examine whether the value depends on price per meter, number of rooms, floor (top or bottom), or the proximity to the downtown area. Also study the correlation to the ad posting date: day of the week, month, and year. "Select the 10 places with the largest number of ads and then calculate the average price per square foot in these localities. Select the locations with the highest and lowest housing prices. You can find this data by name in the ’*locality_name’* column. "
# - [ ]  task completed: "Thoroughly look at apartment offers: each apartment has information about the distance to the downtown area. Select apartments in Saint Petersburg (*‘locality_name’*). Your task is to pinpoint which area is considered to be downtown. Create a column with the distance to the downtown area in km and round to the nearest whole number. Next, calculate the average price for each kilometer. Build a graph to display how prices are affected by the distance to the downtown area. Define the turning point where the graph significantly changes. This will indicate downtown. "
# - [ ]  task completed: "Select a segment of apartments in the downtown. Analyze this area and examine the following parameters: square area, price, number of rooms, ceiling height. Also identify the factors that affect an apartment’s price (number of rooms, floor, distance to the downtown area, and ad publication date). Draw your conclusions. Are they different from the overall conclusions about the entire city?"
# - [ ]  each stage has a conclusion
# - [ ]  overall conclusion drawn
