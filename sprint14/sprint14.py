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
# <div class="alert alert-warning">
#     If I have some (optional) suggestions, or questions to think about, or general comments.
# </div>
# 
# <div class="alert alert-danger">
#     If a section requires some corrections. Work can't be accepted with red comments.
# </div>
# 
# Please don't remove reviewer's comments! All iterations of one project are done by the same reviewer, and if you remove comments from the previous iteration, it makes it hard to understand what changed and what problems there were. Thankfully, as I recall, your project didn't have any problems, apart from a bit unclear conclusions, and you fixed that. I hope you don't mind if I don't reproduce all the comments praising your work :) The project is accepted. Good luck on the next sprint!

# # Statement

# The Film Junky Union, a new edgy community for classic movie enthusiasts, is developing a system for filtering and categorizing movie reviews. The goal is to train a model to automatically detect negative reviews. You'll be using a dataset of IMBD movie reviews with polarity labelling to build a model for classifying positive and negative reviews. It will need to have an F1 score of at least 0.85.

# # Init

# In[5]:


get_ipython().run_cell_magic('capture', '', '%%bash\n\npip install transformers -qq\npip install contractions\npip install emoji\npip install catboost -q\npython -m spacy download en -q\npython -m spacy download en_core_web_sm -q\npip install gdown -q')


# In[6]:


import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import *
import sklearn.metrics as metrics
import seaborn as sns
from tqdm.auto import tqdm
import string
import gc
import nltk
nltk.download('stopwords')  
from nltk.corpus import stopwords

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import spacy

import contractions
import emoji
from html import unescape
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import *
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.dummy import DummyClassifier

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset, RandomSampler, SequentialSampler, random_split
# import torchtext
from tqdm.notebook import tqdm
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
import transformers
from transformers import BertTokenizer, BertConfig, BertModel, AdamW, get_linear_schedule_with_warmup
import time
import h5py
import gensim
import gensim.downloader
from collections import defaultdict
import random
import warnings
warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


# In[7]:


def setup_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

random_state=42
setup_seed(random_state)


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png'")
# the next line provides graphs of better quality on HiDPI screens
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
plt.style.use('seaborn')


# In[9]:


# this is to use progress_apply, read more at https://pypi.org/project/tqdm/#pandas-integration
tqdm.pandas()


# In[10]:


get_ipython().run_cell_magic('capture', '', '\n!pip install ipython-autotime\n \n%load_ext autotime')


# In[7]:


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


# In[ ]:


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


# In[ ]:


def to_h5(in_object, outfile):
  with h5py.File('data.h5', 'w') as h5f:
    h5f.create_dataset('dataset_1', data=in_object)

def from_h5(in_file):
  with h5py.File('data.h5','r') as h5f:
     return h5f['dataset_1'][:]


# # Load Data

# In[ ]:


df_reviews = pd.read_csv('https://code.s3.yandex.net/datasets/imdb_reviews.tsv', sep='\t', dtype={'votes': 'Int64'})


# In[ ]:


df_reviews.head()


# In[ ]:


df_reviews.info()


# In[ ]:


df_reviews[df_reviews["average_rating"].isna()]


# couple records are missed, it's not big deal, let's remove it

# In[ ]:


df_reviews = df_reviews[~df_reviews["average_rating"].isna()]


# In[ ]:


sns.countplot(df_reviews["pos"])


# as we see, in our case dataset is pretty balanced

# In[ ]:


describe_full(df_reviews)


# # EDA

# Let's check the number of movies and reviews over years.

# In[ ]:


fig, axs = plt.subplots(2, 1, figsize=(16, 8))

ax = axs[0]

dft1 = df_reviews[['tconst', 'start_year']].drop_duplicates()     ['start_year'].value_counts().sort_index()
dft1 = dft1.reindex(index=np.arange(dft1.index.min(), max(dft1.index.max(), 2021))).fillna(0)
dft1.plot(kind='bar', ax=ax)
ax.set_title('Number of Movies Over Years')

ax = axs[1]

dft2 = df_reviews.groupby(['start_year', 'pos'])['pos'].count().unstack()
dft2 = dft2.reindex(index=np.arange(dft2.index.min(), max(dft2.index.max(), 2021))).fillna(0)

dft2.plot(kind='bar', stacked=True, label='#reviews (neg, pos)', ax=ax)

dft2 = df_reviews['start_year'].value_counts().sort_index()
dft2 = dft2.reindex(index=np.arange(dft2.index.min(), max(dft2.index.max(), 2021))).fillna(0)
dft3 = (dft2/dft1).fillna(0)
axt = ax.twinx()
dft3.reset_index(drop=True).rolling(5).mean().plot(color='orange', label='reviews per movie (avg over 5 years)', ax=axt)

lines, labels = axt.get_legend_handles_labels()
ax.legend(lines, labels, loc='upper left')

ax.set_title('Number of Reviews Over Years')

fig.tight_layout()


# Let's check the distribution of number of reviews per movie with the exact counting and KDE (just to learn how it may differ from the exact counting)

# In[ ]:


fig, axs = plt.subplots(1, 2, figsize=(16, 5))

ax = axs[0]
dft = df_reviews.groupby('tconst')['review'].count()     .value_counts()     .sort_index()
dft.plot.bar(ax=ax)
ax.set_title('Bar Plot of #Reviews Per Movie')

ax = axs[1]
dft = df_reviews.groupby('tconst')['review'].count()
sns.kdeplot(dft, ax=ax)
ax.set_title('KDE Plot of #Reviews Per Movie')

fig.tight_layout()


# In[ ]:


df_reviews['pos'].value_counts()


# In[ ]:


fig, axs = plt.subplots(1, 2, figsize=(12, 4))

ax = axs[0]
dft = df_reviews.query('ds_part == "train"')['rating'].value_counts().sort_index()
dft = dft.reindex(index=np.arange(min(dft.index.min(), 1), max(dft.index.max(), 11))).fillna(0)
dft.plot.bar(ax=ax)
ax.set_ylim([0, 5000])
ax.set_title('The train set: distribution of ratings')

ax = axs[1]
dft = df_reviews.query('ds_part == "test"')['rating'].value_counts().sort_index()
dft = dft.reindex(index=np.arange(min(dft.index.min(), 1), max(dft.index.max(), 11))).fillna(0)
dft.plot.bar(ax=ax)
ax.set_ylim([0, 5000])
ax.set_title('The test set: distribution of ratings')

fig.tight_layout()


# Distribution of negative and positive reviews over the years for two parts of the dataset

# In[ ]:


fig, axs = plt.subplots(2, 2, figsize=(16, 8), gridspec_kw=dict(width_ratios=(2, 1), height_ratios=(1, 1)))

ax = axs[0][0]

dft = df_reviews.query('ds_part == "train"').groupby(['start_year', 'pos'])['pos'].count().unstack()
dft.index = dft.index.astype('int')
dft = dft.reindex(index=np.arange(dft.index.min(), max(dft.index.max(), 2020))).fillna(0)
dft.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('The train set: number of reviews of different polarities per year')

ax = axs[0][1]

dft = df_reviews.query('ds_part == "train"').groupby(['tconst', 'pos'])['pos'].count().unstack()
sns.kdeplot(dft[0], color='blue', label='negative', kernel='epa', ax=ax)
sns.kdeplot(dft[1], color='green', label='positive', kernel='epa', ax=ax)
ax.legend()
ax.set_title('The train set: distribution of different polarities per movie')

ax = axs[1][0]

dft = df_reviews.query('ds_part == "test"').groupby(['start_year', 'pos'])['pos'].count().unstack()
dft.index = dft.index.astype('int')
dft = dft.reindex(index=np.arange(dft.index.min(), max(dft.index.max(), 2020))).fillna(0)
dft.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('The test set: number of reviews of different polarities per year')

ax = axs[1][1]

dft = df_reviews.query('ds_part == "test"').groupby(['tconst', 'pos'])['pos'].count().unstack()
sns.kdeplot(dft[0], color='blue', label='negative', kernel='epa', ax=ax)
sns.kdeplot(dft[1], color='green', label='positive', kernel='epa', ax=ax)
ax.legend()
ax.set_title('The test set: distribution of different polarities per movie')

fig.tight_layout()


# # Evaluation Procedure

# Composing an evaluation routine which can be used for all models in this project

# In[ ]:


def evaluate_model(model, train_features, train_target, test_features, test_target):
    
    eval_stats = {}
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6)) 
    
    for type, features, target in (('train', train_features, train_target), ('test', test_features, test_target)):
        
        eval_stats[type] = {}
    
        pred_target = model.predict(features)
        pred_proba = model.predict_proba(features)[:, 1]
        
        # F1
        f1_thresholds = np.arange(0, 1.01, 0.05)
        f1_scores = [metrics.f1_score(target, pred_proba>=threshold) for threshold in f1_thresholds]
        
        # ROC
        fpr, tpr, roc_thresholds = metrics.roc_curve(target, pred_proba)
        roc_auc = metrics.roc_auc_score(target, pred_proba)    
        eval_stats[type]['ROC AUC'] = roc_auc

        # PRC
        precision, recall, pr_thresholds = metrics.precision_recall_curve(target, pred_proba)
        aps = metrics.average_precision_score(target, pred_proba)
        eval_stats[type]['APS'] = aps
        
        if type == 'train':
            color = 'blue'
        else:
            color = 'green'

        # F1 Score
        ax = axs[0]
        max_f1_score_idx = np.argmax(f1_scores)
        ax.plot(f1_thresholds, f1_scores, color=color, label=f'{type}, max={f1_scores[max_f1_score_idx]:.2f} @ {f1_thresholds[max_f1_score_idx]:.2f}')
        # setting crosses for some thresholds
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(f1_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(f1_thresholds[closest_value_idx], f1_scores[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('threshold')
        ax.set_ylabel('F1')
        ax.legend(loc='lower center')
        ax.set_title(f'F1 Score') 

        # ROC
        ax = axs[1]    
        ax.plot(fpr, tpr, color=color, label=f'{type}, ROC AUC={roc_auc:.2f}')
        # setting crosses for some thresholds
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(roc_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'            
            ax.plot(fpr[closest_value_idx], tpr[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(loc='lower center')        
        ax.set_title(f'ROC Curve')
        
        # PRC
        ax = axs[2]
        ax.plot(recall, precision, color=color, label=f'{type}, AP={aps:.2f}')
        # setting crosses for some thresholds
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(pr_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(recall[closest_value_idx], precision[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.legend(loc='lower center')
        ax.set_title(f'PRC')        

        eval_stats[type]['Accuracy'] = metrics.accuracy_score(target, pred_target)
        eval_stats[type]['F1'] = metrics.f1_score(target, pred_target)
    
    df_eval_stats = pd.DataFrame(eval_stats)
    df_eval_stats = df_eval_stats.round(2)
    df_eval_stats = df_eval_stats.reindex(index=('Accuracy', 'F1', 'APS', 'ROC AUC'))
    
    print(df_eval_stats)
    
    return


# # Normalization

# We assume all models below accepts texts in lowercase and without any digits, punctuations marks etc.

# In[ ]:


def clean_text(sentence):
  return " ".join(t for t in sentence.split() if not t.isdigit() and t.isalpha())


def clean_column(in_data:pd.Series) -> np.ndarray:
  df = in_data.str.strip().str.lower().str.strip(string.punctuation+" ")
  df = df.str.replace("[^a-zA-Z]", " ")
  df = np.vectorize(contractions.fix)(df)
  df = np.vectorize(emoji.demojize)(df)
  df = np.vectorize(unescape)(df)
  df = np.vectorize(clean_text)(df)
  return df


# In[11]:


import os
if os.path.exists('/content/drive/My Drive/imdb.csv.zip'):
  df_reviews = pd.read_csv('/content/drive/My Drive/imdb.csv.zip')
else:
  df_reviews['review_norm'] = clean_column(df_reviews['review'])
  df_reviews.to_csv("imdb.csv", index=False)
  try:
    get_ipython().system('zip imdb.csv.zip imdb.csv')
    get_ipython().system('cp imdb.csv.zip /content/drive/My\\ Drive/')
  except:
    pass

df_reviews['review_norm'].head(2)


# # Train / Test Split

# Luckily, the whole dataset is already divided into train/test one parts. The corresponding flag is 'ds_part'.

# In[12]:


df_reviews_train = df_reviews.query('ds_part == "train"').copy()
df_reviews_test = df_reviews.query('ds_part == "test"').copy()

train_target = df_reviews_train['pos']
test_target = df_reviews_test['pos']

print(df_reviews_train.shape)
print(df_reviews_test.shape)


# In[13]:


target = "pos"
features = list(set(df_reviews_train.columns)-set([target])-set(["review"]))


# In[ ]:


X_train, X_test, y_train, y_test =  df_reviews_train[features], df_reviews_test[features], train_target, test_target


# >also in real case need split X_train to train/valid sets

# # Model 0 - Constant

# In[ ]:


model = DummyClassifier().fit(X_train, y_train)
y_pred = model.predict(X_test)
display_classification_report(y_test, y_pred)


# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(8,4))
ax = ax.flatten()
_ = plot_pr(y_test, y_pred, ax=ax[0],label="DummyClassifier")
_ = plot_roc(y_test, y_pred, ax=ax[1],label="DummyClassifier")


# In[ ]:


model = DummyClassifier().fit(X_train, y_train)
evaluate_model(model, X_train, y_train, X_test, y_test)


# just random guessing 50/50

# # Model 1 BoW + Naive Bayes

# let's try classic approach: CountVectorizer(BoW)+ Naive Bayes

# In[ ]:


vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(df_reviews_train['review_norm'].values)
classifier = MultinomialNB()
targets = df_reviews_train['pos'].values
classifier = classifier.fit(counts, targets)


# In[ ]:


y_pred = classifier.predict(vectorizer.transform(df_reviews_test['review_norm'].values))
display_classification_report(y_test, y_pred)
fig, ax = plt.subplots(1,2, figsize=(8,4))
ax = ax.flatten()
_ = plot_pr(y_test, y_pred, ax=ax[0],label="Naive Bayes")
_ = plot_roc(y_test, y_pred, ax=ax[1],label="Naive Bayes")


# In[ ]:


evaluate_model(classifier, counts, targets, vectorizer.transform(df_reviews_test['review_norm'].values), y_test)


# as we see, for a reason classic Naive Bayes was popular, without any problem we got pretty good result, which is overfitted but it's not a problem for that case

# In[ ]:


del classifier, counts, targets


# # Model 2 - NLTK, TF-IDF and LR

# ### TF-IDF

# let's try insted of counts, to use more sophisticated tf-idf + simplest neural network;) -> Logistic Regression, but for simplicity I am not going to use cross validation as it was on the previous model

# also I use ngram (for rid of typos)

# In[ ]:


tfv=TfidfVectorizer(min_df=0, max_features=10_000, strip_accents='unicode',lowercase =True,
                            analyzer='word', token_pattern=r'\w{3,}', ngram_range=(1,1),
                            use_idf=True,smooth_idf=True, sublinear_tf=True, stop_words = "english")   
X_train_tfv = tfv.fit_transform(df_reviews_train['review_norm'].values)


# In[ ]:


lr = LogisticRegression().fit(X_train_tfv, y_train)
y_pred = lr.predict(tfv.transform(df_reviews_test['review_norm'].values))
display_classification_report(y_test, y_pred)
fig, ax = plt.subplots(1,2, figsize=(8,4))
ax = ax.flatten()
_ = plot_pr(y_test, y_pred, ax=ax[0],label="LR+Tf-Idf")
_ = plot_roc(y_test, y_pred, ax=ax[1],label="LR+Tf-Idf")


# In[ ]:


evaluate_model(lr, X_train_tfv, y_train, tfv.transform(df_reviews_test['review_norm'].values), y_test)


# I believe we got better result in this case, seems that figures are closed to CountVectoriser and Naive Bayes model, but it's more well-considered result, that is not so overfitted as previous therefore numbers are more confidential

# In[ ]:


kf = KFold(n_splits=5,random_state=random_state, shuffle=True)
f1_cv = cross_val_score(lr, X_train_tfv, y_train, cv=kf, scoring='f1')
f1_cv.mean()


# ### Reviews

# In[ ]:


texts = pd.Series([
      'I did not simply like it, not my kind of movie.',
      'Well, I was bored and felt asleep in the middle of the movie.',
      'I was really fascinated with the movie',    
      'Even the actors looked really old and disinterested, and they got paid to be in the movie. What a soulless cash grab.',
      'I didn\'t expect the reboot to be so good! Writers really cared about the source material',
      'The movie had its upsides and downsides, but I feel like overall it\'s a decent flick. I could see myself going to see it again.',
      'What a rotten attempt at a comedy. Not a single joke lands, everyone acts annoying and loud, even kids won\'t like this!',
      'Launching on Netflix was a brave move & I really appreciate being able to binge on episode after episode, of this exciting intelligent new drama.'])

def show_test_proba(model, data):
    my_reviews_pred_prob = model.predict_proba(data)[:, 1]
    for i, review in enumerate(texts.str.slice(0, 100)):
      print(f'{my_reviews_pred_prob[i]:.2f}:  {review}')


# In[ ]:


show_test_proba(lr, data=tfv.transform(clean_column(texts)).todense())


# # Model 3 - spaCy, TF-IDF and LR

# In[ ]:


get_ipython().system("nvidia-smi | grep 'CUDA Version'")


# In[ ]:


import spacy
spacy.__version__


# In[ ]:


# python -m spacy download en_core_web_sm
# python -m spacy download en

%%capture
%%bash

pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.4/en_core_web_sm-2.2.4.tar.gz#egg=en_core_web_sm -U


# In[ ]:


spacy.prefer_gpu()


# In[ ]:


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


# In[ ]:


def array_preprocessing_3(X):
  nlp_pipes = nlp.pipe(X, disable=["parser", "ner"], n_threads=4, batch_size=1000) 
  return np.array([' '.join(token.lemma_ for token in doc) for doc in nlp_pipes])


# In[ ]:


def text_preprocessing_3(text):
    return ' '.join(token.lemma_ for token in nlp(text))


# In[ ]:


X_train_tfv = array_preprocessing_3(df_reviews_train["review"].values)
tfv=TfidfVectorizer(min_df=0, max_features=10_000, strip_accents='unicode',lowercase =True,
                            analyzer='word', token_pattern=r'\w{3,}', ngram_range=(1,1),
                            use_idf=True,smooth_idf=True, sublinear_tf=True, stop_words = "english")   
X_train_tfv = tfv.fit_transform(X_train_tfv)


# In[ ]:


X_test_tfv = array_preprocessing_3(df_reviews_test["review"].values)
X_test_tfv = tfv.transform(X_test_tfv)


# In[ ]:


lr = LogisticRegression().fit(X_train_tfv, y_train)
y_pred = lr.predict(X_test_tfv)
display_classification_report(y_test, y_pred)
fig, ax = plt.subplots(1,2, figsize=(8,4))
ax = ax.flatten()
_ = plot_pr(y_test, y_pred, ax=ax[0],label="LR+Tf-Idf+lemmas")
_ = plot_roc(y_test, y_pred, ax=ax[1],label="LR+Tf-Idf+lemmas")


# In[ ]:


evaluate_model(lr, X_train_tfv, y_train, X_test_tfv, y_test)


# ### Reviews

# In[ ]:


show_test_proba(lr, data=tfv.transform(np.vectorize(text_preprocessing_3)(pd.Series(clean_column(texts)))).todense())


# # Model 4 - spaCy, TF-IDF and LGBMClassifier

# In[ ]:


model = LGBMClassifier().fit(X_train_tfv, y_train)
y_pred = model.predict(X_test_tfv)
display_classification_report(y_test, y_pred)
fig, ax = plt.subplots(1,2, figsize=(8,4))
ax = ax.flatten()
_ = plot_pr(y_test, y_pred, ax=ax[0],label="LGBMClassifier+Tf-Idf+lemmas")
_ = plot_roc(y_test, y_pred, ax=ax[1],label="LGBMClassifier+Tf-Idf+lemmas")


# In[ ]:


evaluate_model(model, X_train_tfv, y_train, X_test_tfv, y_test)


# ### Reviews

# In[ ]:


show_test_proba(model, data=tfv.transform(np.vectorize(text_preprocessing_3)(pd.Series(clean_column(texts)))).todense())


# # Model 5 RandomForestClassifier

# In[ ]:


skl_model = RandomForestClassifier(n_estimators=10, max_depth=10).fit(X_train_tfv, y_train)
evaluate_model(skl_model, X_train_tfv, y_train, X_test_tfv, y_test)


# by default random forest shows poor result, the key is hypertuning, cv and etc, (which is time expensive)

# In[ ]:


y_pred = skl_model.predict(X_test_tfv)
display_classification_report(y_test, y_pred)
fig, ax = plt.subplots(1,2, figsize=(8,4))
ax = ax.flatten()
_ = plot_pr(y_test, y_pred, ax=ax[0],label="Naive Bayes")
_ = plot_roc(y_test, y_pred, ax=ax[1],label="Naive Bayes")


# #  Model 9 - BERT

# In[14]:


from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
torch.backends.cudnn.benchmark = True

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")


# First of all let's check possible max length

# In[ ]:


def len_tokens(s):
  return len(s.split())

df = df_reviews_train.copy()
df["len"] = np.vectorize(len_tokens)(df["review_norm"])
print("max len is: {}".format(max(list(map(len, tokenizer.batch_encode_plus(df.sort_values(by="len", ascending=False)[:1]["review_norm"].to_list())["input_ids"])))))
sns.displot(df["len"])
del df


# in real case we could create embeddings by chunk 512 and pool it (meaning or by some another way) but in this case we see, that according dist of length, sentences that are greater than 512, actually are not so important. but let's think that max_length=128 in our case for fast computation (bert has O(n^2) complexity depends on length of input)

# In[15]:


max_length = 128
default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu_device = torch.device('cpu')
bert_path="bert-base-uncased"
batch_size=32


# In[16]:


class TokenizersDataset(Dataset):

    def __init__(self, in_data, targets, tokenizer, max_len, splitter_func=None):
        self.data = in_data
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.splitter_func = splitter_func
        assert len(in_data)==len(targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch = [self.data[idx]]
        is_pretokenized = self.splitter_func is not None
        if is_pretokenized:
            batch = [self.splitter_func(sentence) for sentence in batch]

        encoding = self.tokenizer.encode_plus(
            batch,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            is_pretokenized=is_pretokenized,
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'target' : self.targets[idx]
        }

    @staticmethod
    def from_data(data, targets, tokenizer, max_len, batch_size=8):
        ds = TokenizersDataset(
            in_data=data,
            targets=targets,
            tokenizer=tokenizer,
            max_len=max_len,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            # num_workers=2, -> on colab i have only one GPU
            # pin_memory=True
        )


# In[17]:


train_dataloader = TokenizersDataset.from_data(df_reviews_train["review_norm"].values, df_reviews_train["pos"].values, tokenizer, max_len=max_length, batch_size=batch_size)
test_dataloader = TokenizersDataset.from_data(df_reviews_test["review_norm"].values, df_reviews_test["pos"].values, tokenizer, max_len=max_length, batch_size=batch_size)


# In[20]:


class SentimentClassifier(nn.Module):
  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-cased')
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)


# In[49]:


n_examples=2
model = SentimentClassifier(n_examples)
EPOCHS = 10
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)


# In[53]:


def accuracy(outputs, labels):
    predicted = torch.argmax(outputs, dim=1)
    return torch.mean(torch.eq(predicted, labels).float()).item()


# In[60]:


def train_epoch(model,data_loader,loss_fn,optimizer,device,scheduler,n_examples, epoch):
      pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}. Train Loss: {0}")
      device = default_device
      model.to(device)
      model = model.train()
      loss_fn = nn.CrossEntropyLoss().to(device)
      losses = []
      correct_predictions = 0
      for step, d in enumerate(pbar):
          targets = d["target"].to(device)
          outputs = model(
            input_ids=d["input_ids"].to(device),
            attention_mask=d["attention_mask"].to(device)
          ).to(device)
          _, preds = torch.max(outputs, dim=1)
          loss = loss_fn(outputs, targets)
          correct_predictions += torch.sum(preds == targets)
          losses.append(loss.item())
          loss.backward()
          nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
          optimizer.step()
          scheduler.step()
          optimizer.zero_grad()
          
          acc = accuracy(outputs, targets)
          pbar.set_description(f"Epoch:{epoch + 1}.Train Loss:{loss:.4} Acc:{acc:.4}")
          if step%100==0:gc.collect();torch.cuda.empty_cache();
          
          
      return correct_predictions.double() / n_examples, np.mean(losses)


# In[ ]:


from collections import defaultdict
history = defaultdict(list)
best_accuracy = 0
for epoch in range(EPOCHS):
  print(f'Epoch {epoch + 1}/{EPOCHS}')
  print('-' * 10)
  train_acc, train_loss = train_epoch(model,train_dataloader,loss_fn,optimizer,default_device,scheduler,len(df_reviews_train), epoch)
  print(f'Train loss {train_loss} accuracy {train_acc}')


# # Conclusions

# Several models were checked and different methods: linear models, using lemmatization, a bag of words, TF-IDF, and transformers, BERT
# faster and good according to F1-score was BoW models with Naive Bayes (also I think we could use instead of this model, e.g. xgboost model) 
# some smart model with BERT approach didn't show some significant result, but due to some limitation of the resources I didn't check good model(was the only simple model) on it, without searching hyperparameters and others optimizations
# 
# 

# BoW + Naive Bayes
# 
# <div class="stream"><div class="output_subarea output_text"><pre>          train  test
# Accuracy   0.90  0.81
# F1         0.89  0.80
# APS        0.96  0.87
# ROC AUC    0.96  0.89
# </pre></div></div>
# 
# 
# NLTK, TF-IDF and LR
# 
# 
# <pre>          train  test
# Accuracy   0.93  0.88
# F1         0.93  0.88
# APS        0.98  0.95
# ROC AUC    0.98  0.95
# </pre>
# 
# 
# spaCy, TF-IDF and LR
# 
# 
# <div class="stream"><div class="output_subarea output_text"><pre>          train  test
# Accuracy   0.92  0.88
# F1         0.92  0.88
# APS        0.98  0.95
# ROC AUC    0.98  0.95
# </pre></div></div>
# 
# 
# spaCy, TF-IDF and LGBMClassifier
# 
# 
# <pre>          train  test
# Accuracy   0.91  0.85
# F1         0.91  0.86
# APS        0.97  0.93
# ROC AUC    0.97  0.93
# </pre>

# as we see best result by test and f1 score, we got using TfIdf + LR which is make sense, in this case, sentiment analysis

# also was finetuned bert model, but due limitation of resources I didn't train-evaluate model properly and therefore didn't consider it on result stage

# # Check List

# - [x]  Notebook was opened
# - [ ]  The text data is loaded and pre-processed for vectorization
# - [ ]  The text data is transformed to vectors
# - [ ]  Models are trained and tested
# - [ ]  The metric's threshold is reached
# - [ ]  All the code cells are arranged in the order of their execution
# - [ ]  All the code cells can be executed without errors
# - [ ]  There are conclusions

# In[ ]:




