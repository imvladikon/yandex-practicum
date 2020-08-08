#!/usr/bin/env python
# coding: utf-8

# ## Commets from reviewer
# 
# <span style="color:green"> Hi! Congratulations on your first project :)You did a great job here! I really enjoyed your project, it's very deep and detailed. There is nothing that I can add to your work) Good luck with the next one! </span>

# ## Analyzing borrowers’ risk of defaulting
# 
# Your project is to prepare a report for a bank’s loan division. You’ll need to find out if a customer’s marital status and number of children has an impact on whether they will default on a loan. The bank already has some data on customers’ credit worthiness.
# 
# Your report will be considered when building a **credit scoring** of a potential customer. A ** credit scoring ** is used to evaluate the ability of a potential borrower to repay their loan.

# ### Step 1. Open the data file and have a look at the general information. 

# In[647]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# ><h2><strong>Describing data</strong></h2><div class="paragraph">—<em>children</em> : the number of children in the family </div><div class="paragraph">—<em>days_employed</em>: how long the customer has worked </div><div class="paragraph">—<em>dob_years</em>: the customer’s age </div><div class="paragraph">—<em>education</em>: the customer’s education level </div><div class="paragraph">—<em>education_id</em>: identifier for the customer’s education </div><div class="paragraph">—<em>family_status</em>: the customer’s marital status </div><div class="paragraph">—<em>family_status_id</em>: identifier for the customer’s marital status </div><div class="paragraph">—<em>gender</em>: the customer’s gender </div><div class="paragraph">—<em>income_type</em>: the customer’s income type </div><div class="paragraph">—<em>debt</em>: whether the client has ever defaulted on a loan </div><div class="paragraph">—<em>total_income</em>: monthly income </div><div class="paragraph">—<em>purpose</em>: reason for taking out a loan</div><h2>

# In[648]:


borrowers_df = pd.read_csv('/datasets/credit_scoring_eng.csv')
borrowers_df.head()
borrowers_df.tail()
borrowers_df.sample(5)


# In[649]:


pd.concat([borrowers_df.dtypes, borrowers_df.isna().sum()], axis=1)


# In[650]:


borrowers_df.describe(include='all')


# In[651]:


borrowers_df.info()


# In[652]:


#it's not days, we have huge gap, mean is 183 years 
(abs(borrowers_df["days_employed"])/365).describe()


# In[653]:


borrowers_df[borrowers_df['days_employed'] < 0]['days_employed'].count()
borrowers_df[borrowers_df['days_employed'] >= 0]['days_employed'].apply(lambda x: x / 365).describe()


# In[654]:


borrowers_df[(abs(borrowers_df['days_employed']) <= 75*365)]['days_employed'].count()


# Only negative value of days_employed are less than max of dobs_year

# It seems that we have incorrect data in the column 'days_employed'. I assume that problem was appeared after saving in Excel (format problem). In Excel date format, as I know, uses ole datetime (in the C# we have DateTime.FromOADate method), https://docs.microsoft.com/en-us/dotnet/api/system.datetime.fromoadate?view=netframework-4.8  we can write the same method on the python. Also it could be epoch time since 1970 (problem is also unknown measure of our column. we have max=401755 and min=-18388, it is unlikely that our column in days, because diff between min and max in this case more than 1000 years)

# In[655]:


from datetime import datetime, timedelta
import numpy as np
from math import isnan

OLE_TIME_ZERO = datetime(1899, 12, 30, 0, 0, 0)

def from_ole(ts:float)->datetime:
    if isnan(ts):
        return ts
    return OLE_TIME_ZERO + timedelta(days=float(ts/10))
def from_epoch(ts:float):
    if isnan(ts):
        return ts
    return datetime.fromtimestamp(ts*3600)
borrowers_df["days_employed"].min()
borrowers_df["days_employed"].max()
from_ole(borrowers_df["days_employed"].min())
from_ole(borrowers_df["days_employed"].max())
from_epoch(borrowers_df["days_employed"].min())
from_epoch(borrowers_df["days_employed"].max())
days = abs(borrowers_df["days_employed"])/abs(borrowers_df["days_employed"].min())
days.describe()


# In[656]:


borrowers_df["date_employed"] = borrowers_df.apply(lambda row: from_ole(abs(row["days_employed"])), axis=1)


# In[657]:


borrowers_df[["date_employed", "dob_years"]].sample(15)


# In[658]:


borrowers_df["date_employed"] = borrowers_df.apply(lambda row: from_epoch(abs(row["days_employed"])), axis=1)
borrowers_df[["date_employed", "dob_years"]].sample(5)


# As we see, idea with date_employed didn't work

# In[659]:


borrowers_df = borrowers_df.drop('date_employed', axis=1)
borrowers_df.head()


# In[660]:


duplicates = borrowers_df[borrowers_df.duplicated()]
duplicates.sample(5)
100*len(duplicates)/len(borrowers_df)


# ### Conclusion

# * $\approx$ 10% of data in columns "days_employed", "income_type" is missed 
# * days_employed" is incorrect column, it couldn't be negative, and it has unknown measure. we need fix it or drop this column
# * we need specify one case for string (object) columns (for example: lower)
# * we need categorize data and lemmatize (column "purpose")
# * we need specify type of "days_employed" and "income_type"
# 
# * Dataset has 54 (0.3%) duplicate rows	
# * days_employed has 2174 (10.1%) missing values	
# * total_income has 2174 (10.1%) missing values
# * some information has the same meaning like family_status and family_status_id

# children - min=-1 and max=20, it's impossible
# days_employed is wrong column, max is 401755, for days it's more than 1100years
# dob_years - min=0 is incorrect

# * total_income - Numeric
# * purpose - Categorical
# * income_type - Categorical
# * gender - boolean
# * family_status Categorical (family_status_id Numeric)
# * education Categorical (education_id Numeric)
# * dob_years Numeric
# * debt Boolean
# * days_employed Numeric
# * children Numeric

# <span style="color:green"> Very nice and deep approaches!

# ### Step 2. Data preprocessing

# ### Processing missing values

# In[661]:


total = borrowers_df.isnull().sum().sort_values(ascending=False)
percent = (borrowers_df.isnull().sum()/borrowers_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Missing Percent'])
missing_data['Missing Percent'] = missing_data['Missing Percent'].apply(lambda x: x * 100)
missing_data.loc[missing_data['Missing Percent'] > 10][:10]


# ~10% of missing data is a lot. let's see, there is a relation between missed data and other columns

# In[662]:


df_nan = borrowers_df[borrowers_df['total_income'].isna()]
df_nan.groupby('education_id')['education_id'].count()


# In[663]:


borrowers_df[borrowers_df['days_employed'].isnull()].sample(10)


# As wee see, we have ~10% data around 'total_income' and 'days_employed' at the same time. In my opinion it's hard to assume what data was missed and it's hard fairly interpolate 'total_income' and 'days_employed', but it's quantitative variables, missing values in quantitative variables are filled with representative values like average values. Logically correct to fill it depends on income_type, let's see

# In[664]:


borrowers_df[borrowers_df['days_employed'].isna()].pivot_table(index='income_type',values='total_income' ,aggfunc='count')


# In[665]:


median_income_type = borrowers_df.groupby('income_type').agg({'total_income': 'mean'}).reset_index()
median_income_type


# In[666]:


replacer_map = {row["income_type"]:row["total_income"] for i, row in median_income_type.iterrows()}
borrowers_df['total_income'] = borrowers_df['total_income'].fillna(borrowers_df["income_type"].map(replacer_map))
borrowers_df[borrowers_df['days_employed'].isnull()].sample(5)


# In[667]:


borrowers_df[borrowers_df['days_employed'].isnull()]['total_income'].value_counts()


# In[668]:


borrowers_df[borrowers_df['days_employed'].isnull()]['debt'].value_counts()


# In[669]:


def fill_days_employed():
    global borrowers_df
    fillna_value("days_employed", borrowers_df['days_employed'].median())
def fillna_value(column, value):
    global borrowers_df
    borrowers_df[column].fillna(value, inplace=True)


# In[670]:


borrowers_df = borrowers_df[pd.notnull(borrowers_df['days_employed'])]
borrowers_df.head()


# In[671]:


borrowers_df.describe(include='all')


# as we see we have invalid values in the children's column - -1, 20. (max and min). assume, that -1 is 1 and 20 is 2, probably someone mistyped
# les's correct it

# In[672]:


borrowers_df['children'] = borrowers_df['children'].replace(-1, 1)
borrowers_df['children'] = borrowers_df['children'].replace(20, 2)
borrowers_df['days_employed']= abs(borrowers_df['days_employed'])
borrowers_df['gender'].unique()
borrowers_df[borrowers_df['gender']=='XNA']


# there is incorrect record with XNA gender. let's see, do we have other records with the same income_type

# In[673]:


income_type = borrowers_df[borrowers_df['gender']=='XNA'].reset_index()['income_type'][0]
borrowers_df[borrowers_df['income_type'] == income_type].shape
borrowers_df.groupby('income_type').agg(['mean', 'count'])


# There are 5085 records with 'partner' income_type. And we have only 1 record with incorrect gender value. 
# We could delete this record

# In[674]:


borrowers_df = borrowers_df[borrowers_df['gender'] != 'XNA']
borrowers_df[borrowers_df['gender']=='XNA']


# ### Conclusion

# As wee see, we have ~10% data around 'total_income' and 'days_employed' at the same time. In my opinion it's hard to assume what data was missed and it's hard fairly interpolate 'total_income' and 'days_employed', it's quantitative Variables, missing values in quantitative variables are filled with representative values. 

# ### Data type replacement

# In[675]:


borrowers_df.info()


# In[676]:


borrowers_df['days_employed'].sample(5)


# In[677]:


borrowers_df['total_income'].sample(5)


# In[678]:


borrowers_df['total_income'].dtypes


# In[679]:


borrowers_df['days_employed'] = borrowers_df['days_employed'].astype(int, errors='ignore')
borrowers_df['total_income'] = borrowers_df['total_income'].astype(int, errors='ignore')


# In[680]:


borrowers_df['total_income'].dtypes


# ### Conclusion

# We converted days_employed and total_income from float to int type, to optimize computing and memory usage

# ### Processing duplicates

# In[681]:


borrowers_df['education'].value_counts()


# In[682]:


def to_case():
    global borrowers_df
    borrowers_df['education'] = borrowers_df['education'].str.lower()
    borrowers_df['family_status'] = borrowers_df['family_status'].str.lower()
    borrowers_df['gender'] = borrowers_df['gender'].str.upper()
    borrowers_df['income_type'] = borrowers_df['income_type'].str.lower()
    borrowers_df['purpose'] = borrowers_df['purpose'].str.lower()


# In[683]:


to_case()


# In[684]:


borrowers_df.duplicated().sum()
borrowers_df[borrowers_df.duplicated()].sort_values('education').head()


# In[685]:


borrowers_df = borrowers_df.drop_duplicates().reset_index(drop=True)
borrowers_df.duplicated().sum()


# ### Conclusion

# Drop duplicates is logically correct when we have differences only in string cases. Also we have duplicates by meaning in "purpose" column. That's why we need to do lemmatizaion

# <span style="color:green"> Great job with preprocessing)

# ### Lemmatization

# In[686]:


purposes = borrowers_df['purpose'].unique()
display(purposes)
len(purposes)


# In[687]:


import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
stopwords = set(stopwords.words('english'))
is_noun = lambda pos: pos[:2] == 'NN'


# In[688]:


wordnet_lemma = WordNetLemmatizer()
lemmas = []
for text in purposes:
    words = nltk.word_tokenize(text.lower())
    lemmas += [wordnet_lemma.lemmatize(w, pos='n') for (w,pos) in nltk.pos_tag(words) if w not in stopwords and is_noun(pos)]
purpose_categories = list(Counter(lemmas).keys())
purpose_categories


# In[689]:


from nltk.stem import SnowballStemmer 
english_stemmer = SnowballStemmer('english')


# In[690]:


category_purpose_dict = {text_purpose:next(category for category in purpose_categories if english_stemmer.stem(category) in text_purpose or category in text_purpose) for text_purpose in set(borrowers_df['purpose'])}
category_purpose_dict


# In[691]:


#some hand fixing
for k, v in category_purpose_dict.items():
    if not v=='car' and 'car' in k:
        category_purpose_dict[k] = 'car'
        continue
    if not v=='house' and ('estate' in k or 'house' in k):
        category_purpose_dict[k] = 'house'
        continue
    if not v=='property' and ('construction' in k or 'property' in k):
        category_purpose_dict[k] = 'property'
        continue
category_purpose_dict


# In[692]:


borrowers_df['purpose'] = borrowers_df['purpose'].replace(category_purpose_dict)
borrowers_df.sample(5)


# ### Conclusion

# For our purposes(for example - classification) we need to process our data (purpose column) which is represented by sentences on natural language.  Therefore column of purpose has been lemmatized, it will be easier categorize our data in the future 

# <span style="color:green"> Lemmatization is done nicely

# ### Categorizing Data

# * purpose - Categorical
# * income_type - Categorical
# * gender - Categorical
# * family_status - Categorical
# * education - Categorical

# In[693]:


to_case()


# In[694]:


borrowers_df.drop_duplicates(['family_status_id', 'family_status'])[['family_status_id', 'family_status']]


# In[695]:


borrowers_df.drop_duplicates(['education_id', 'education'])[['education_id', 'education']]


# As we see, there is no problem with education_id, education and family_status_id, family_status, those columns have identical length and meaning, there is no gap. We could encode categorical columns or just use id for this purpose

# In[696]:


map_purpose = {purpose:index for index, purpose in enumerate(borrowers_df['purpose'].unique())}
map_purpose
map_gender = {'F':1,'M':0}
map_income_type = {income_type:index for index, income_type in enumerate(borrowers_df['income_type'].unique())}
map_income_type
map_family_status = {row['family_status_id']:row['family_status'] for _, row in borrowers_df.drop_duplicates(['family_status_id', 'family_status'])[['family_status_id', 'family_status']].iterrows()}
map_family_status
map_education = {row['education_id']:row['education'] for _, row in borrowers_df.drop_duplicates(['education_id', 'education'])[['education_id', 'education']].iterrows()}
map_education


# In[697]:


# alternative method is LabelEncoder
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
def process_income_type():
    global borrowers_df
    borrowers_df['income_type_id'] = borrowers_df['income_type'].map(map_income_type)
def process_purpose():
    global borrowers_df
    borrowers_df["purpose_id"] = borrowers_df["purpose"].map(map_purpose)
#     borrowers_df['purpose'] = le.fit_transform(borrowers_df['purpose'])
def process_gender():
    global borrowers_df
    borrowers_df['gender_id'] = borrowers_df['gender'].map(map_gender)
    borrowers_df['gender_id'] = borrowers_df['gender_id'].astype(int)
#     borrowers_df["gender"] = borrowers_df["gender"].astype('category')
#     borrowers_df["gender"] = borrowers_df["gender"].cat.codes
def delete_columns():
    global borrowers_df
    cols = ['income_type', 'gender', 'purpose', 'education', 'family_status']
    borrowers_df = borrowers_df.drop([col for col in cols if col in borrowers_df.columns], axis=1)


# In[698]:


process_income_type()
process_purpose()
process_gender()
borrowers_df.head()


# In[699]:


delete_columns()
borrowers_df.head()


# In[700]:


borrowers_df['total_income'].describe()


# In[701]:


borrowers_df.plot.scatter(x='income_type_id',
                      y='total_income',
                      c= 'total_income',
                      colormap='viridis')
borrowers_df.plot.scatter(x='purpose_id',
                      y='total_income',
                      c= 'total_income',
                      colormap='viridis')


# In[702]:


borrowers_df[['dob_years', 'total_income', 'children']].describe()

def category_total_income(total_income):
    if total_income < 75000:
        return 0
    elif 75000 <= total_income < 120000:
        return 1
    elif 120000 <= total_income < 150000:
        return 2
    elif 150000 <= total_income < 200000:
        return 3
    else:
        return 4
    
def category_children(children):
    if children < 1:
        return 0
    elif 1 <= children < 3:
        return 1
    else:
        return 2

def category_dob_years(dob_years):
    if dob_years < 35:
        return 0
    elif 35 <= dob_years < 45:
        return 1
    else:
        return 2
    
def process_categorize():
    global borrowers_df 
    borrowers_df['children_category'] = borrowers_df['children'].apply(category_children)
    borrowers_df['total_income_category'] = borrowers_df['total_income'].apply(category_total_income)
    borrowers_df['dob_years_category'] = borrowers_df['dob_years'].apply(category_dob_years)
    


# In[703]:


process_categorize()
borrowers_df.head()


# ### Conclusion

# Categorization is important task, and we categorized 'dob_years', 'total_income' and 'children'columns. Purpose, gender, family_status, education were encoded. It's important step in the EDA, in the future we could use this columns for example for building some models(decision tree).

# <span style="color:green"> This step is correct

# ### Step 3. Answer these questions

# - Is there a relation between having kids and repaying a loan on time?

# In[704]:


borrowers_df['has_children'] = borrowers_df['children'].apply(lambda c: int(c>0))
borrowers_df_children = borrowers_df[borrowers_df['debt']==0].groupby(['children','debt']).size().reset_index().merge(borrowers_df[borrowers_df['debt']==1].groupby(['children','debt']).size().reset_index(), on='children')
borrowers_df_children["rate"] = 100*borrowers_df_children["0_y"]/borrowers_df_children["0_x"]
borrowers_df_children


# In[705]:


borrowers_df_children = borrowers_df[borrowers_df['debt']==0].groupby(['children_category','debt']).size().reset_index().merge(borrowers_df[borrowers_df['debt']==1].groupby(['children_category','debt']).size().reset_index(), on='children_category')
borrowers_df_children["rate"] = 100*borrowers_df_children["0_y"]/borrowers_df_children["0_x"]
borrowers_df_children


# as we could see, there is a difference between borrowers that having kids and repaying a loan on time, 
# but it's not like that in relative figures, because we see an equal distribution 

# In[706]:


plt.scatter(borrowers_df['debt'], borrowers_df['children'])
plt.show()


# Let's see Spearman correlation. The sign of the Spearman correlation indicates the direction of association between X (the independent variable) and Y (the dependent variable). If Y tends to increase when X increases, the Spearman correlation coefficient is positive. If Y tends to decrease when X increases, the Spearman correlation coefficient is negative. A Spearman correlation of zero indicates that there is no tendency for Y to either increase or decrease when X increases. The Spearman correlation increases in magnitude as X and Y become closer to being perfectly monotone functions of each other.

# In[707]:


borrowers_df[['children_category', 'debt']].corr(method='spearman')
borrowers_df[['children', 'debt']].corr(method='spearman')


# ### Conclusion

# A relation between having kids and repaying a loan on time is weak. We tried to find some correlation using Spearman correlation. People without kids have less problems with repaying a loan on time

# - Is there a relation between marital status and repaying a loan on time?

# In[708]:


borrowers_df_family = borrowers_df[borrowers_df['debt']==0].groupby(['family_status_id','debt']).size().reset_index().merge(borrowers_df[borrowers_df['debt']==1].groupby(['family_status_id','debt']).size().reset_index(), on='family_status_id')
borrowers_df_family["rate"] = 100*borrowers_df_family["0_y"]/borrowers_df_family["0_x"]
borrowers_df_family["family_status"] = borrowers_df_family["family_status_id"].map(map_family_status)
borrowers_df_family


# In[709]:


borrowers_df[['family_status_id', 'debt']].corr(method='spearman')


# ### Conclusion

# We have the same picture like before. There is weak relation, it looks like widow / widower and divorced groups have good figures. 

# - Is there a relation between income level and repaying a loan on time?

# In[710]:


plt.scatter(borrowers_df['total_income'], borrowers_df['debt'])
plt.show()


# In[711]:


map_total_income = {0:'<75000', 1:'[75000, 120000)',2:'[120000, 150000)',3:'[150000,200000)',4:'>=200000'}
borrowers_df_income = borrowers_df[borrowers_df['debt']==0].groupby(['total_income_category','debt']).size().reset_index().merge(borrowers_df[borrowers_df['debt']==1].groupby(['total_income_category','debt']).size().reset_index(), on='total_income_category')
borrowers_df_income["rate"] = 100*borrowers_df_income["0_y"]/borrowers_df_income["0_x"]
borrowers_df_income["total_income_title"] = borrowers_df_income["total_income_category"].map(map_total_income)
borrowers_df_income


# In[712]:


borrowers_df[['total_income_category', 'debt']].corr(method='spearman')


# ### Conclusion

# There is a weak negative correlation, but it close to zero. Also we could notice, clients with <75000 and >=200000 total_income have best figures 

# - How do different loan purposes affect on-time repayment of the loan?

# In[713]:


borrowers_df.head()


# In[714]:


map_purpose_reverse = {v:k for k, v in map_purpose.items()}
borrowers_df_purpose = borrowers_df[borrowers_df['debt']==0].groupby(['purpose_id','debt']).size().reset_index().merge(borrowers_df[borrowers_df['debt']==1].groupby(['purpose_id','debt']).size().reset_index(), on='purpose_id')
borrowers_df_purpose["rate"] = 100*borrowers_df_purpose["0_y"]/borrowers_df_purpose["0_x"]
borrowers_df_purpose["purpose"] = borrowers_df_purpose["purpose_id"].map(map_purpose_reverse)
borrowers_df_purpose


# In[715]:


borrowers_df[['purpose_id', 'debt']].corr(method='spearman')


# ### Conclusion

# There is weak relation between purpose of loan and debt. As we saw, a majority of debtors are from 'car' and 'education' groups

# ### Step 4. General conclusion

# It's important to analyze data and find out some dependencies and correlations for building a highly accurate predictive algorithm that could predict the creditworthiness of the clients. However, in general, no correlation has been identified. There is a weak correlation between having children, also purposes of loans could impact the picture of the credit score in general speaking. It can be assumed that there is correlation between multiple features at the same time. Also we could describe portrait of good client, - without kids, with total income more than 200000, and with purposes on loan    

# <span style="color:green"> Step 3 and Step 4 are great! Congratulations!

# ### Project Readiness Checklist
# 
# Put 'x' in the completed points. Then press Shift + Enter.

# - [x]  file open;
# - [ ]  file examined;
# - [ ]  missing values defined;
# - [ ]  missing values are filled;
# - [ ]  an explanation of which missing value types were detected;
# - [ ]  explanation for the possible causes of missing values;
# - [ ]  an explanation of how the blanks are filled;
# - [ ]  replaced the real data type with an integer;
# - [ ]  an explanation of which method is used to change the data type and why;
# - [ ]  duplicates deleted;
# - [ ]  an explanation of which method is used to find and remove duplicates;
# - [ ]  description of the possible reasons for the appearance of duplicates in the data;
# - [ ]  highlighted lemmas in the values of the loan purpose column;
# - [ ]  the lemmatization process is described;
# - [ ]  data is categorized;
# - [ ]  an explanation of the principle of data categorization;
# - [ ]  an answer to the question "Is there a relation between having kids and repaying a loan on time?";
# - [ ]  an answer to the question " Is there a relation between marital status and repaying a loan on time?";
# - [ ]   an answer to the question " Is there a relation between income level and repaying a loan on time?";
# - [ ]  an answer to the question " How do different loan purposes affect on-time repayment of the loan?"
# - [ ]  conclusions are present on each stage;
# - [ ]  a general conclusion is made.

# In[ ]:




