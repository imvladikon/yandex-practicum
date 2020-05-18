#!/usr/bin/env python
# coding: utf-8

# ## Project description

# <div class="paragraph">You work for the online store Ice, which sells videogames all over the world. User and expert reviews, genres, platforms (e.g. Xbox or PlayStation), and historical data on game sales are available from open sources. You need to identify patterns that determine whether a game succeeds or not. This allows you to put your money on a potentially hot new item and plan advertising campaigns.</div><div class="paragraph">In front of you is data going back to 2016. Let’s imagine that it’s December 2016 and you’re planning a campaign for 2017. </div><div class="paragraph">The important thing is to get experience working with data. It doesn't really matter whether you're forecasting 2017 sales based on data from 2016 or 2027 sales based on data from 2026.</div><div class="paragraph">The data set contains the abbreviation <em>ESRB (</em>Entertainment Software Rating Board<em>)</em>. The ESRB evaluates a game's content and assigns an appropriate age categories, such as Teen and Mature.</div>

# In[391]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f as f_test
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


# ## 1. Open the data file and study the general information

# <div class="markdown markdown_size_normal markdown_type_theory theory-viewer__markdown"><h3>Data description</h3><div class="paragraph">—<em>Name</em> </div><div class="paragraph">—<em>Platform</em> </div><div class="paragraph">—<em>Year_of_Release</em> </div><div class="paragraph">—<em>Genre</em> </div><div class="paragraph">—<em>NA_sales</em> (North American sales in USD million) </div><div class="paragraph">—<em>EU_sales</em> (sales in Europe in USD million) </div><div class="paragraph">—<em>JP_sales</em> (sales in Japan in USD million) </div><div class="paragraph">—<em>Other_sales</em> (sales in other countries in USD million) </div><div class="paragraph">—<em>Critic_Score</em> (maximum of 100) </div><div class="paragraph">—<em>User_Score</em> (maximum of 10) </div><div class="paragraph">—<em>Rating</em> (ESRB)</div><div class="paragraph">Data for 2016 may be incomplete.</div></div>

# In[392]:


datafile = "https://code.s3.yandex.net/datasets/games.csv"


# In[393]:


df = pd.read_csv(datafile)


# In[394]:


df.info()


# In[395]:


df.isna().sum()


# ## 2. Prepare the data

# <ul><li>Replace the column names (make them lowercase).</li><li>Convert the data to the required types.</li><li>Describe the columns where the data types have been changed and why.</li><li>If necessary, decide how to deal with missing values:
#   <ul><li>Explain why you filled in the missing values as you did or why you decided to leave them blank.</li><li>Why do you think the values are missing? Give possible reasons.</li><li>Pay attention to the abbreviation TBD (to be determined) in the rating column. Specify how you intend to handle such cases.</li></ul></li><li>Calculate the total sales (the sum of sales in all regions) for each game and put these values in a separate column.</li></ul>

# In[396]:


df.columns = df.columns.str.lower()


# In[397]:


df.dtypes


# In[398]:


df


# In[399]:


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

def plt_missing_data(df):
    data_null = df.isnull().sum()/len(df)
    data_null = data_null.drop(data_null[data_null == 0].index).sort_values(ascending=False)
    plt.subplots(figsize=(40,10))
    plt.xticks(rotation='90')
    sns.barplot(data_null.index, data_null)
    plt.xlabel('Features', fontsize=20)
    plt.ylabel('Missing rate', fontsize=20)


# In[400]:


missing_values(df)


# In[401]:


plt_missing_data(df)


#  as we see we have 6 columns with missing value, user_score and rating have almost the same, genre and name are also

# In[402]:


describe_full(df).T


# as we could see, _sales columns are strongly skew

# In[403]:


col_sales = ['na_sales', 'eu_sales', 'eu_sales', 'jp_sales', 'other_sales']
df["all_sales"] = df[col_sales].sum(axis=1)


# In[404]:


sns.boxplot(df["na_sales"])


# In[405]:


sns.boxplot(df["eu_sales"])


# In[406]:


sns.boxplot(df["jp_sales"])


# In[407]:


df["platform"].unique()


# I think it's better to convert year from float to int, for simplicity of comparing and etc. (we could also convert to datetime, but in this case it's not matter) But first of all we need to deal with impute missing values

# In[408]:


df_missing_year = df[:][df["year_of_release"].isna()].reset_index().drop_duplicates()
df_missing_year


# I didn't see any pattern of missing year. 

# In[409]:


df[df["name"]=="Madden NFL 2004"]


# In[410]:


df[df["name"]=="Space Invaders"]


# In[411]:


df_missing_year[["name"]].drop_duplicates()


# In[412]:


map_name_year = {r["name"]:r["year_of_release"] for _, r in (df_missing_year[["name"]]
 .drop_duplicates()
 .merge(df[~df["year_of_release"].isna()][["name", "year_of_release"]], how="inner").drop_duplicates()).iterrows()}


# In[413]:


df["year_of_release"] = df["year_of_release"].fillna(df["name"].map(map_name_year))


# In[414]:


df[df["year_of_release"].isna()]


# we filled only part of missing values of the years, by name. let's fill the remaining by the minimum. (worse case) but before need to check minimum (as we saw before, we have strange year as minimum)

# In[415]:


df[df["year_of_release"]==df["year_of_release"].min()]


# I assume it's also missing values. Also there are missing values at the 3rd columns - critic_score	user_score	rating. as we see, sales columns have low values of saling. Let's drop this column

# In[416]:


df = df.drop(df[df["year_of_release"]==df["year_of_release"].min()].index)


# In[417]:


sns.distplot(df["year_of_release"])


# we have skewness, and according to business model, data that is before 2000, has uncertainty impact on current (2016) picture. Also there are a lot of missing values around this years and it's logically correct 

# In[418]:


df[(df["year_of_release"]<1995) & (df["all_sales"]<10)]


# If necessary, decide how to deal with missing values:
# Explain why you filled in the missing values as you did or why you decided to leave them blank.
# Why do you think the values are missing? Give possible reasons.
# Pay attention to the abbreviation TBD (to be determined) in the rating column. Specify how you intend to handle such cases.
# Calculate the total sales (the sum of sales in all regions) for each game and put these values in a separate column.

# In[419]:


df[df["genre"].isna()]


# It's also incorrect rows, it's old rows, don't have name, and a lot of columns. we couldn't impute it in a right way

# In[420]:


df = df.drop(df[df["genre"].isna()].index)


# In[421]:


df.isna().sum()


# In[422]:


df.loc[df["year_of_release"].isna(), "year_of_release"] = df["year_of_release"].min()


# In[423]:


sns.distplot(df["all_sales"])


# In[424]:


df[df["critic_score"].isna()]


# I think best score it's sales;)

# In[425]:


df.loc[df["critic_score"].isna(), "critic_score"] = df[df["critic_score"].isna()]["all_sales"].astype(int)


# In[426]:


df["user_score"].unique()


# In[427]:


df["rating"].unique()


# In[428]:


df["year_of_release"] = df["year_of_release"].astype(int)


# as we see we have "to be determined" user score. it's similar to nan in our case and for our model we could replace it by negative 1

# In[429]:


df['user_score'] = df['user_score'].fillna(-1)
df['user_score'] = df['user_score'].replace('tbd', -1)
df['critic_score'] = df['critic_score'].fillna(-1)


# In[430]:


df['user_score'] = df['user_score'].astype('float')


# In[431]:


df["genre"] = df["genre"].str.lower()
# df["genre"] = df["genre"].astype("category")


# we don't know rating of missied row, let's create another category for that. also because there are a lot of rows like that

# * EC - Early childhood
# * E - Everyone
# * E10+ - Everyone 10 and older
# * T -Teen
# * M - Mature
# * AO - Adults Only 18+ 
# * RP - Rating Pending 
# 

# In[432]:


df.loc[df["rating"].isna(), "rating"] = "NA"


# In[433]:


df.duplicated().sum()


# ### Summary

# * We found missing values. Missing values by years were restored by name of games partially and some of them were filled by min value
# * Missing values of users and critics were replaced by negative value, we couldn't delete them and we couldn't fill them by some value logically
# * Also types of year, uuser_score, critic_score were converted to numerical types, and genre was converted as category 
# 

# ## 3. Analyze the data

# <ul><li>Look at how many games were released in different years. Is the data for every period significant?</li><li>Look at how sales varied from platform to platform. Choose the platforms with the greatest total sales and build a distribution based on data for each year. Find platforms that used to be popular but now have zero sales. How long does it generally take for new platforms to appear and old ones to fade?</li><li>Determine what period you should take data for<strong>.</strong> To do so, look at your answers to the previous questions. The data should allow you to build a prognosis for 2017.</li><li>Work only with the data that you've decided is relevant. Disregard the data for previous years.</li><li>Which platforms are leading in sales? Which ones are growing or shrinking? Select several potentially profitable platforms.</li><li>Build a box plot for the global sales of each game, broken down by platform. Are the differences in sales significant? What about average sales on various platforms? Describe your findings.</li><li>Take a look at how user and professional reviews affect sales for a particular popular platform. Build a scatter plot and calculate the correlation between reviews and sales. Draw conclusions.</li><li>Keeping your conclusions in mind, compare the sales of the same games on other platforms.</li><li>Take a look at the general distribution of games by genre. What can we say about the most profitable genres? Can you generalize about genres with high and low sales?</li></ul>

# In[434]:


with plt.style.context('bmh'): 
    ((df[['name', 'year_of_release']]
        .drop_duplicates()
        .pivot_table(index='year_of_release', values='name', aggfunc='count')
        .sort_values('year_of_release', ascending=False))
        .plot(figsize=(6, 3), colormap='jet', legend=False, title='Total games by year')
        .set(xlabel='year', ylabel='count'))


# as we predicted before, and already saw, data before 1995-2000 is not interested for us, it's old games. also it's incorrect from economic view, cost of money was changed. also we could see how trends were changed. probably after 2008-2010 people play more in online game which as I think, are not represented in the dataset 

# In[333]:


((df.pivot_table(index='platform', values='all_sales', aggfunc='sum')
         .sort_values('all_sales', ascending=False))
   .plot(kind='bar', y='all_sales', figsize=(12, 8), legend=False)
            .set(xlabel='platform', ylabel='sum of sales'))
plt.show()


# In[334]:


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


# let's find most saling platform

# In[335]:


top10_platforms = (df
 .pivot_table(index='platform', values='all_sales', aggfunc='sum')
 .sort_values('all_sales', ascending=False)).head(10)
top10_platforms


# and plot density plot for them

# In[336]:


display_group_density_plot(df[df["platform"].isin(top10_platforms.index)], groupby = "platform", on = 'year_of_release',                                            palette = sns.color_palette('Set2'), 
                           figsize = (10, 5))


# as we see some of platforms were popular in different time of perid. and non of top 10 platforms by saling were not popular on period before 1995

# In[337]:


df = df[(df['year_of_release'] > 1995) & (df['year_of_release'] <= 2016)]


# In[338]:


all_sales_per_platform =df.pivot_table(index='platform', values='all_sales', aggfunc='sum').sort_values('all_sales', ascending=False)


# In[339]:


platforms = list(all_sales_per_platform.index)


# let's plot sales for different platforms

# In[340]:


fig, axs = plt.subplots(6, len(platforms) // 6)
axs = axs.flatten()

num = 0
for platform, ax in zip(platforms, axs):
     ((df[df["platform"]==platform]
            .pivot_table(index='year_of_release', values='all_sales', aggfunc='sum')
            .sort_values('year_of_release', ascending=False))['all_sales']
                .plot(ax=ax, figsize=(18, 18), title=platform)
                .set(xlabel='year_of_release', ylabel='all sales'))
   
    
plt.tight_layout()
plt.show()


# according plots, as we assumed before a lot of sales was decreased. some of platforms like WS are gone. "2600" it's something strange
# PS4 is popular comaring to others.

# ## 4. Create a user profile for each region

# <div class="paragraph">For each region (NA, EU, JP), determine:</div><ul><li>The top five platforms. Describe variations in their market shares from region to region.</li><li>The top five genres. Explain the difference.</li><li>Do ESRB ratings affect sales in individual regions?</li></ul>

# In[341]:


def plot_top_sales_by_region_barplot(df, column, region, n=5):
    data = (df
            .pivot_table(index=column, values=region, aggfunc='sum')
            .sort_values(region, ascending=False)
            .head(n)).reset_index()
    sns.set_color_codes("pastel")
    sns.barplot(y=data[column], x=data[region], data=data, label="sales")
    sns.set_color_codes("muted")
    plt.show()


# In[368]:


def plot_top_sales_by_region_pie(df, column, region, n=5):
    data = (df
            .pivot_table(index=column, values=region, aggfunc='sum')
            .sort_values(region, ascending=False)
            .head(n))
    data.plot(kind='pie', y=region,autopct='%1.0f%%', figsize=(10, 5), legend=True).set(label=data.index)
    plt.show()


# #### NA / North America

# In[343]:


plot_top_sales_by_region_barplot(df, 'platform', region = 'na_sales')


# In the USA X360 is popular, second one is PS2. both of company have long story of competition at the USA market, Microsoft v Sony

# In[370]:


plot_top_sales_by_region_pie(df, 'genre', region = 'na_sales')


# as we see most popular genre is an Action

# #### EU / Europe

# In[119]:


plot_top_sales_by_region(df, 'platform', region = 'eu_sales')


# In[372]:


plot_top_sales_by_region_pie(df, 'genre', region = 'eu_sales')


# similar picture to the USA, action and Sports. 

# #### JP / Japan

# In[373]:


plot_top_sales_by_region(df, 'platform', region = 'jp_sales')


# In[375]:


plot_top_sales_by_region_pie(df, 'genre', region = 'jp_sales')


# in the Japan situation is different, RPG is most popular

# ### Summary

# in the USA:
#     * In the USA X360 is popular, second one is PS2. both of company have long story of competition at the USA market, Microsoft v Sony
#     * Action/Sports and Shooter
# in the Europe:
#     * PS2/PS3
#     * genre - action and Sports.
# in the Japan:
#     * DS and PS are popular console
#     * RPG is the most popular genre
#     

# ## 5. Test the following hypotheses:

# In[451]:


#I think it could vary depends on how much data we deleted before, because it could impact on statistics
alpha = .05 


# <div class="paragraph">—Average user ratings of the Xbox One and PC platforms are the same. </div><div class="paragraph">—Average user ratings for the Action and Sports genres are different.</div><div class="paragraph">Set the <em>alpha</em> threshold value yourself.</div><div class="paragraph">Explain:</div><div class="paragraph">—How you formulated the null and alternative hypotheses </div><div class="paragraph">—What significance level you chose to test the hypotheses, and why</div>

# Let's check:
# —Average user ratings of the Xbox One and PC platforms are the same.
# 
# $H_0$:—Average user ratings of the Xbox One and PC platforms are the same.
# 
# and
# 
# $H_1$: —Average user ratings of the Xbox One and PC platforms are different.   
# 

# In[455]:


df["platform"].unique()


# In[462]:


H_0 = "Average user ratings of the Xbox One and PC platforms are the same."
H_1 = "Average user ratings of the Xbox One and PC platforms are different."

pc = df[df["platform"] == 'PC'].dropna()['user_score'].values
xbox = df[df["platform"].isin(["X360", "XOne", "XB"])].dropna()['user_score'].values

result = stats.ttest_ind(pc, xbox)
print('pvalue:', result.pvalue)

if (result.pvalue < alpha):
    print(H_1)
else:
    print(H_0)
print("checking:")
print(f"pc mean={pc.mean()}")
print(f"xbox mean={xbox.mean()}")


# Let's check:
# —Average user ratings of the Xbox One and PC platforms are the same.
# 
# $H_0$:—Average user ratings of the Xbox One and PC platforms are the same.
# 
# and
# 
# $H_1$: —Average user ratings for the Action and Sports genres are different.   
# 

# In[464]:


H_0 = "Average user ratings for the Action and Sports genres are the same."
H_1 = "Average user ratings for the Action and Sports genres are different."

action = df[df["genre"] == 'action'].dropna()['user_score'].values
sports = df[df["genre"] == 'sports'].dropna()['user_score'].values

result = stats.ttest_ind(action, sports)
print('pvalue:', result.pvalue)

if (result.pvalue < alpha):
    print(H_1)
else:
    print(H_0)
print("checking:")
print(f"action mean={action.mean()}")
print(f"sports mean={sports.mean()}")


# ## 6. Write a general conclusion

# * Platforms are before 2005, was gone. Others are still in the market but also in the process of disappearing.
# * Most popular are PlayStation and Xbox. Most popular genre are Action, Shooter, except Japan, where RPG is most popular
# * As we saw it's region specific business. 
# * Average user ratings of the Xbox One and PC platforms are different.
# * Average user ratings for the Action and Sports genres are different.
# * Also need reserch specific platform, big gap between most popular platform and the others
