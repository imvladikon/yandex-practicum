#!/usr/bin/env python
# coding: utf-8

# ## Review (3)
# 
# Now everything is cool, so I'm just accepting your project. Good luck with future learning.
# 
# ---

# ## Review (2)
# 
# Two of three issues are closed, but we still need to work with rounding of minutes and traffic. You did it in fair way :) but mobile operators prefer to round everything up :). I left you a comment inside the notebook with some extra explanation.
# 
# ---

# ## Review
# 
# Hi Vladimir. My name is Soslan. I'm reviewing your work. I've added all my comments to new cells with the title "Review". My apologies for the delay in the review. We will be faster next time :)
# 
# ```diff
# + If you did something great I'm using green color for my comment
# - If the topic requires some extra work so I can accept it then the color will be red.
# ```
# 
# You did most of the work. Your code is very good quality and works correctly. But you need a bit more work with textual content and conclusions. Can you please look through my comments and work with them so I could accept your work. Good luck.

# In[107]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.core.interactiveshell import InteractiveShell
import re
from scipy import stats as stats
from scipy.stats import f as f_test
InteractiveShell.ast_node_interactivity = 'all'
import warnings
warnings.filterwarnings('ignore')


# # Open the data file and study the general information

# <div class="paragraph">You work as an analyst for "Megaline", a state mobile operator. The company offers its clients two prepaid plans, Surf and Ultimate. The commercial department would like to know which of the plans is more profitable in order to adjust the advertising budget. </div><div class="paragraph">You are going to carry out a preliminary analysis of the plans based on a relatively small client selection. You'll have the data on 500 "Megaline" clients, specifically, who the clients are, where they are from, which plan they use, the number of calls made and SMS they sent in 2018. You have to analyse clients' behavior and work out the most profitable prepaid plan. </div><h3>Prepaid plans description</h3><div class="paragraph"><strong>Surf</strong></div>
# 
# <ol start="1"><li>Monthly charge: \$20</li><li>500 monthly minutes, 50 SMS and 15 GB of web traffic</li><li>After exceeding the package limits:
#  1. 1 minute: 3 cents ("Megaline" always rounds up the minute and megabyte values. If the call lasted just one second, it will be counted as one minute);
#  2. SMS: 3 cents;
#  3. 1 GB of web traffic: \$10.</li></ol><div class="paragraph"><strong>Ultimate</strong></div><ol start="1"><li>Monthly charge: $70</li><li>3000 monthly minutes, 1000 SMS and 30 GB of web traffic</li><li>After exceeding the package limits:
#  1. 1 minute: 1 cent;
#  2. SMS: 1 cent;
#  3. 1 GB of web traffic: \$7.</li></ol>

# 
# <div class="markdown markdown_size_normal markdown_type_theory"><h2><strong>Project description</strong></h2><h3>Data description</h3><div class="paragraph">The <code class="code-inline code-inline_theme_light">users</code> table (data on users):</div><ul><li><em>user_id</em> — unique user identifier</li><li><em>first_name</em> — user's name</li><li><em>last_name</em> — user's last name</li><li><em>age</em> — user's age (years)</li><li><em>reg_date</em> — subscription date (dd, mm, yy)</li><li><em>churn_date</em> — the date of use discontinue (if the value is missed, the calling plan was used at the moment of data extraction)</li><li><em>city</em> — user's city of residence</li><li><em>tarif</em> — calling plan name</li></ul><div class="paragraph">The <code class="code-inline code-inline_theme_light">calls</code> table (data on calls):</div><ul><li><em>id</em> — unique call identifier</li><li><em>call_date</em> — call date</li><li><em>duration</em> — call duration in minutes</li><li><em>user_id</em> — the identifier of the user making a call</li></ul><div class="paragraph">The <code class="code-inline code-inline_theme_light">messages</code> table (data on SMS):</div><ul><li><em>id</em> — unique SMS identifier</li><li><em>message_date</em> — SMS date</li><li><em>user_id</em> — the identifier of the user sending an SMS</li></ul><div class="paragraph">The <code class="code-inline code-inline_theme_light">internet</code> table (data on web sessions):</div><ul><li><em>id</em> — unique session id</li><li><em>mb_used</em> —  the volume of web traffic spent during a session (in megabytes)</li><li><em>session_date</em> — web session date</li><li><em>user_id</em> — user identifier</li></ul><div class="paragraph">The <code class="code-inline code-inline_theme_light">tariffs</code> table (data on the plans):</div><ul><li><em>tariff_name</em> — calling plan name</li><li><em>rub_monthly_fee</em> — monthly charge in rubles</li><li><em>minutes_included</em> — monthly minutes within package limits</li><li><em>messages_included</em> — monthly SMS within package limits</li><li><em>mb_per_month_included</em> — web traffic volume within package limits (in megabytes)</li><li><em>rub_per_minute</em> — the price per minute after exceeding the package limits (e.g., if the package included 100 minutes, the 101st minute will be charged)</li><li><em>rub_per_message</em> — the price per SMS after exceeding the package limits</li><li><em>rub_per_gb</em> — the price per extra gigabyte of web traffic after exceeding the package limits (1 GB = 1024 megabytes)</li></ul></div>

# In[155]:


host="https://code.s3.yandex.net/"


# In[156]:


megaline_calls = pd.read_csv(host+"datasets/megaline_calls.csv")
megaline_internet = pd.read_csv(host+"datasets/megaline_internet.csv")
megaline_messages = pd.read_csv(host+"datasets/megaline_messages.csv")
megaline_tariffs = pd.read_csv(host+"datasets/megaline_tariffs.csv")
megaline_users = pd.read_csv(host+"datasets/megaline_users.csv")


# In[157]:


megaline_datasets = {"megaline_calls":megaline_calls, "megaline_internet":megaline_internet, "megaline_messages":megaline_messages, "megaline_tariffs":megaline_tariffs, "megaline_users":megaline_users}


# In[158]:


for name, ds in megaline_datasets.items():
    f"{name} {ds.shape}"


# In[159]:


for name, ds in megaline_datasets.items():
    f"{name}"
    ds.info()


# # Prepare the data

# <ul><li>Convert the data to the necessary types;</li><li>Find and remove the errors in the data.</li></ul><div class="paragraph">Explain what errors did you find and how did you remove them. 
# Please note: quite a lot of calls have a duration of 0.0 minutes, a strong hint that there is a problem with the data and needs preprocessing. </div><div class="paragraph">For each user, find:</div><ul><li>the number of calls made and minutes spent per month;</li><li>the number of SMS sent per month;</li><li>the volume of web traffic per month;</li><li>the monthly profit from each of the users (subtract free package limit from the total number of calls, SMS and web traffic; multiply the result by the calling plan value; add monthly charge depending on the calling plan).</li></ul>

# In[160]:


for name, ds in megaline_datasets.items():
    f"{name}"
    ds.sample(2)


# As we some columns have object(str) type.Let's fix it, convert them

# In[161]:


megaline_calls["call_date"] = pd.to_datetime(megaline_calls["call_date"], format='%Y-%m-%d')


# In[162]:


megaline_internet["session_date"] = pd.to_datetime(megaline_internet["session_date"], format='%Y-%m-%d')


# In[163]:


megaline_messages["message_date"] = pd.to_datetime(megaline_messages["message_date"], format='%Y-%m-%d')


# In[164]:


megaline_users["reg_date"] = pd.to_datetime(megaline_users["reg_date"], format='%Y-%m-%d')


# In[165]:


megaline_users["churn_date"] = pd.to_datetime(megaline_users["churn_date"], format='%Y-%m-%d')


# In[166]:


def missing_values(df):
  df_nulls=pd.concat([df.dtypes, df.isna().sum(), df.isna().sum()/len(df)], axis=1)
  df_nulls.columns = ["type","count","missing_ratio"]
  df_nulls=df_nulls[df_nulls["count"]>0]
  df_nulls.sort_values(by="missing_ratio", ascending=False)
  return df_nulls


# Let's check missing ratio of data in our datasets

# In[167]:


for name, ds in megaline_datasets.items():
  print(name)
  missing_values(ds)


# churn_date — the date of use discontinue (if the value is missed, the calling plan was used at the moment of data extraction)
# 

# In[120]:


megaline_users.columns


# In[168]:


megaline_users.loc[megaline_users["churn_date"].isna(), "churn_date"] = pd.datetime.now().date()


# In[169]:


megaline_users["churn_date"] = pd.to_datetime(megaline_users["churn_date"], format='%Y-%m-%d')


# In[170]:


megaline_users["used_days"] = (megaline_users["churn_date"]-megaline_users["reg_date"]).dt.days


# In[171]:


megaline_users = megaline_users.drop(["churn_date"], axis=1)


# In[172]:


def describe_full(df):
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


# In[173]:


for name, ds in megaline_datasets.items():
  print(name)
  describe_full(ds)


# We have denormalized table, let's build table that could neatly fit for ours purpose

# In[174]:


megaline_users_messages = megaline_users.merge(megaline_messages).rename({"message_id":"id"}, axis=1)


# In[175]:


megaline_users_internet = megaline_users.merge(megaline_internet).rename({"session_id":"id"}, axis=1)


# In[176]:


megaline_users_call = megaline_users.merge(megaline_calls).rename({"call_id":"id"}, axis=1)


# In[177]:


megaline_users_messages['message_month'] = megaline_users_messages['message_date'].dt.month


# In[178]:


megaline_users_internet['session_month'] = megaline_users_internet['session_date'].dt.month


# In[179]:


megaline_users_call['call_month'] = megaline_users_call['call_date'].dt.month


# In[180]:


megaline_users_internet.info()


# In[181]:


megaline_users_internet["mb_used"] = megaline_users_internet["mb_used"].apply(np.ceil) 


# In[182]:


len(megaline_users_internet["id"].unique()) == len(megaline_users_internet["id"].str.replace("_", "").astype(int).unique())


# In[183]:


megaline_users_internet["id"] = megaline_users_internet["id"].str.replace("_", "").astype(int)


# In[184]:


map_tariff = {'ultimate':0, 'surf':1}


# In[185]:


megaline_users_internet.tariff = (megaline_users_internet.tariff == 'surf').astype(int)


# In[186]:


megaline_users_call.info()


# In[187]:


megaline_users_call["duration"] = megaline_users_call["duration"].apply(np.ceil)


# In[188]:


len(megaline_users_call["id"].unique()) == len(megaline_users_call["id"].str.replace("_", "").astype(int).unique())


# In[189]:


megaline_users_call["id"] = megaline_users_call["id"].str.replace("_", "").astype(int)


# In[190]:


megaline_users_call.tariff = (megaline_users_call.tariff == 'surf').astype(int)


# In[191]:


megaline_users_messages.info()


# As we see id has pattern of "number-number" , maybe we could convert it to numeric format,  but before it we need
# to check of validity of removing of "-"

# In[192]:


len(megaline_users_messages["id"].unique()) == len(megaline_users_messages["id"].str.replace("_", "").astype(int).unique())


# In[193]:


megaline_users_messages["id"] = megaline_users_messages["id"].str.replace("_", "").astype(int)


# In[194]:


megaline_users_messages.tariff = (megaline_users_messages.tariff == 'surf').astype(int)


# In[195]:


megaline_users.tariff = (megaline_users.tariff == 'surf').astype(int)


# In[196]:


megaline_datasets = {"megaline_users_messages":megaline_users_messages, "megaline_users_call":megaline_users_call, "megaline_users_internet":megaline_users_internet}


# In[197]:


for name, ds in megaline_datasets.items():
  print(name)
  print(ds.duplicated().sum())


# In[198]:


megaline_users_call_pivot = pd.DataFrame(megaline_users_call.pivot_table(index=['user_id', 'call_month'], values='duration', aggfunc=['count', 'sum'])).reset_index(drop=False)
megaline_users_call_pivot.columns = ['user_id', 'call_month', 'count_calls', 'total_duration']
megaline_users_call_pivot.sample(2)


# As we know Megaline round up duration and internet:

# ## Review (2)
# 
# ```diff
# - If the tail of number is less then .5 method .round() rounds it to the lower int number and if more or equal then to higher. Here you should always round your numbers to a higher value. So if the customer used 1.3 mb for the session company charges him for 2 mb. Same for minutes. Here method np.ceil could help you.
# 
# - You are rounding your data after aggregating it by months but should do it by every session or call and only after then round the data.
# ```
# 
# Here an useful link about how it works: http://www.datasciencemadesimple.com/ceil-floor-dataframe-pandas-python-2/
# 
# ---
# 

# >Thanks, added round up above, removed it from pivot tables

# In[200]:


megaline_users_call_pivot.sample(2)


# In[201]:


megaline_users_messages_pivot = pd.DataFrame(megaline_users_messages.pivot_table(index=['user_id', 'message_month'], values='id', aggfunc=['count'])).reset_index(drop=False)
megaline_users_messages_pivot.columns = ['user_id', 'msg_month', 'count_msg']
megaline_users_messages_pivot.sample(2)


# In[202]:


megaline_users_internet_pivot = pd.DataFrame(megaline_users_internet.pivot_table(index=['user_id', 'session_month'], values='mb_used', aggfunc='sum')).reset_index(drop=False)
megaline_users_internet_pivot.columns = ['user_id', 'session_month', 'total_mb']
megaline_users_internet_pivot.sample(2)


# In[204]:


megaline_agg_df = (megaline_users
  .join(megaline_users_call_pivot.groupby('user_id')[['total_duration', 'count_calls']].sum(), on='user_id')
  .join(megaline_users_internet_pivot.groupby('user_id')['total_mb'].sum(), on='user_id')
  .join(megaline_users_messages_pivot.groupby('user_id')['count_msg'].sum(), on='user_id'))


# In[205]:


#kx+b=y, 500 when tariff is 1 and  3000 when is 0, we have -2500x+3000
megaline_agg_df['duration_left'] = megaline_agg_df["total_duration"] - (-2500*megaline_agg_df["tariff"] + 3000)


# In[206]:


def calc_call_profit(r):
    duration = r['duration_left']
    tariff = 0.03 if r['tariff'] else 0.01
    return duration*tariff


# In[207]:


megaline_agg_df['profit_call'] = megaline_agg_df.apply(calc_call_profit, axis=1)


# In[208]:


megaline_agg_df['mb_left'] = megaline_agg_df["total_mb"] - (-15*1024*megaline_agg_df["tariff"]+30*1024)


# In[209]:


def calc_internet_profit(r):
    internet = r['total_mb']/1024
    tariff = 10 if r['tariff'] else 7
    return internet*tariff


# In[210]:


megaline_agg_df['profit_internet'] = megaline_agg_df.apply(calc_internet_profit, axis=1)


# In[211]:


megaline_agg_df['msg_left'] = megaline_agg_df["count_msg"] - (-950*megaline_agg_df["tariff"] + 1000)


# In[212]:


def calc_msg_profit(r):
    msg = r['msg_left']
    tariff = 0.03 if r['tariff'] else 0.01
    return msg*tariff


# In[213]:


megaline_agg_df['profit_msg'] = megaline_agg_df.apply(calc_msg_profit, axis=1)


# In[214]:


months = 12


# In[215]:


megaline_agg_df['profit_total'] = (megaline_agg_df['profit_internet'] + megaline_agg_df['profit_call'] + megaline_agg_df['profit_msg']) +(-50*megaline_agg_df["tariff"]+70)


# In[216]:


megaline_agg_df['duration_per_month'] = megaline_agg_df['total_duration'] / months
megaline_agg_df['calls_month'] = megaline_agg_df['count_calls'] / months
megaline_agg_df['internet_mb_per_month'] = megaline_agg_df['total_mb'] / months
megaline_agg_df['msg_per_month'] = megaline_agg_df['count_msg'] / months


# In[217]:


megaline_agg_df['profit_per_month'] = megaline_agg_df['profit_total'] / months


# In[218]:


megaline_agg_df.fillna(0)


# ### Summary

# We had really low missed data, - only churn_date, but it's logically correct, cause on the extraction date users still use services. created new feature - used days, - count of days
# in duration of that users used to services and column churn_date was dropped. Also datetimes columns was converted to demanded format. 
# Also features were created for representing the month usage of services.

# ## Review
# 
# ```diff
# + You did more than needed in this section, prepare a lot of new features this is great.
# - You forgot to apply providers trick with rounding up minutes and megabytes ("Megaline" always rounds up the minute and megabyte values. If the call lasted just one second, it will be counted as one minute) can you also implement this :)
# ```
# 
# It is a bit hard to track your work as you leave few comments inside. Your variables names are great, but some accompanying text would be great.

# >Thanks! I fixed it

# # Analyse the data

# <div class="paragraph">Describe the clients' behavior. For the users of each of the plans, find the number of minutes and SMS and the volume of web traffic they require per month. Calculate the mean, variance and standard deviation. Plot histograms. Describe the distributions. </div>

# We are going to calculate this metrics:
# call per month usage, internet per month usage, mb, messages per month usage, profit per month

# In[219]:


metrics = {'calls_month':'call per month usage', 'internet_mb_per_month':'internet per month usage, mb', 'msg_per_month':'messages per month usage', 'profit_per_month':'profit per month'}


# Let's create plot function, for comparing distribution of our metrics comparing for our plans

# In[220]:


def generate_plots(df, metrics=metrics):
   ultimate = df[df["tariff"]==0]
   surf = df[df["tariff"]==1]
   for col, title in metrics.items():
     plt.hist(ultimate[col], alpha=0.5, label=f'ultimate {col}')
     plt.hist(surf[col], alpha=0.5, label=f'surf {col}')
     plt.legend(loc='upper right')
     plt.show()


# In[221]:


generate_plots(megaline_agg_df)


# In[222]:


def generate_report(df, tariff_repr, metrics=metrics):
   print(f"{tariff_repr}") 
   tariff = 0 if tariff_repr == "ultimate" else 1
   dfq = df[df["tariff"]==tariff]
   funcs = {'average':np.mean, 'variance':np.var, 'std':np.std}
   for col, title in metrics.items():
     for method_name, func in funcs.items():
         print(f'{method_name} {title}: {func(dfq[col]):.2f}')


# ### Summary

# In[223]:


generate_report(megaline_agg_df, 'ultimate')


# In[224]:


generate_report(megaline_agg_df, 'surf')


# ## Review
# 
# ```diff
# + Your code is great and compact. Everything works correctly
# - Your work needs more textual content. Some explanations for people who don't understand code. What is happening, what is the results of your analysis, some insights. For example, I see that you calculated monthly profit for users, but I don't see anything about it in your reports.
# ```

# >Fixed it, added some explanations;)!

# # Test the hypotheses

# <ul><li>the average profit from the users of Ultimate and Surf calling plans is different;</li><li>the average profit from the users in NY-NJ area is different from that of the users from other regions.</li></ul><div class="paragraph">Assign the threshold <em>alpha</em> value independently. </div><div class="paragraph">Explain:</div><ul><li>how you formulated the null and alternative hypotheses;</li><li>what criterion you used for testing the hypotheses and why.</li></ul>

# ##### Hypotheses is avg profit of two plans are not different (the null hypothesis that Var(X) == Var(Y))
# We will use F-test
# 
# 

# In[225]:


megaline_agg_df.columns


# In[226]:


ultimate = megaline_agg_df[megaline_agg_df["tariff"] == 0]['profit_per_month'].dropna()
surf = megaline_agg_df[megaline_agg_df["tariff"] == 1]['profit_per_month'].dropna()


# In[227]:


def check_f_test(X,Y):
  F = np.var(X) / np.var(Y)
  p_value = f_test.cdf(F, len(X)-1, len(Y)-1)
  return p_value > .05


# In[228]:


if check_f_test(ultimate, surf):
    print('could not reject the null hypothesis, therefore that Var(ultimate) == Var(surf)')
else:
    print('There is significance to reject the null hypothesis, therefore Var(ultimate) != Var(surf)')


# In[229]:


results = stats.ttest_ind(ultimate, surf, equal_var=True)
print(results.pvalue)
if results.pvalue < .05:
      print('reject the null hypothesis, therefore the average profits for plans are different')
else:
      print('Fail in rejecting of the null hypothesis, therefore the average profits for plans are the same')


# The average profit from the users of Ultimate and Surf calling plans is different;

# Let's handle with NY areay, first of all we need to ckeck all unique value, find the pattern for detection of area

# In[230]:


megaline_agg_df["city"].unique()
# "NY-NJ"


# In[231]:


megaline_agg_df["is_NY"] = megaline_agg_df["city"].str.contains("NY-NJ")


# In[232]:


NY = megaline_agg_df[megaline_agg_df["is_NY"]]['profit_per_month'].dropna()
not_NY = NY = megaline_agg_df[~megaline_agg_df["is_NY"]]['profit_per_month'].dropna()


# In[233]:


if check_f_test(NY, not_NY):
    print('There is significance to reject the null hypothesis, therefore Var(ultimate) == Var(surf)')   
else:
    print('Failed to reject the null hypothesis, therefore Var(ultimate) != Var(surf)')
    


# In[234]:


results = stats.ttest_ind(NY, not_NY, equal_var=True)
if results.pvalue < .05:
      print('we have sufficient significance for rejecting the null hypothesis, therefore the avg profits comparing for two population in NY area or not are different')
        
else:
      print('Fail in rejecting of the null hypothesis, therefore the avg profits comparing for two population in NY area or not are the same')


# # Write an overall conclusion

# <div class="paragraph"><strong>Format:</strong> Complete the task in <em>Jupyter Notebook</em>. Insert the programming code in <em>code</em> cells and text explanations in <em>markdown</em> cells then apply formatting and headings.</div>

# ## Summary

# We checked ultimate and surf plans
# 
# Checked data, some statistics, also dists for some total features were checked
# 
# Also were checked  null hypotesis: average profit of the ultimate plan and the surf plan is the same, (significance level is 0.05)
# 
# The same check we did for NY-NJ clients and other regions
# 

# ## Review
# 
# ```diff
# - And again, code is great, but your conclusions contradict to your code. In the case of different tariffs you reject the null hypothesis, and in the case of different areas - not. But you copypaste your code and don't change text second time so in the overall conclusion, you mess results of your tests.
# ```

# >I apologise for that! I added notes, yes I duplicated code, bit didn't correct it after(

# ## Review (2)
# 
# ```diff
# + In two last section everything is correct now, thanks.
# ```
# 
# ---

# In[ ]:




