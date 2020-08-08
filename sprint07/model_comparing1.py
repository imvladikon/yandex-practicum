#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from joblib import dump


# In[5]:


df = pd.read_csv('./datasets/train_data_us.csv')
df.loc[df['last_price'] > 113000, 'price_class'] = 1
df.loc[df['last_price'] <= 113000, 'price_class'] = 0
features = df.drop(['last_price', 'price_class'], axis=1)
target = df['price_class']


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.2, random_state=42)
len(y_train),len(y_test)


# In[14]:


tree_param = {'n_estimators':list(range(1, 11)),'max_depth':[2, 4, 8, 12], 'min_samples_leaf':[4,8], 'max_leaf_nodes':[7, 9]}


# In[15]:


clf = GridSearchCV(RandomForestClassifier(), tree_param, cv=5, scoring=make_scorer(score_func=score, greater_is_better=True))
clf.fit(X_train, y_train)


# In[17]:


clf.best_params_


# In[18]:


params = {'max_depth': 12,
 'max_leaf_nodes': 9,
 'min_samples_leaf': 8,
 'n_estimators': 10}
model = RandomForestClassifier(**params)
model.fit(X_train, y_train)


# In[19]:


y_pred = model.predict(X_test)


# In[21]:


print(classification_report(y_pred, y_test))


# In[22]:


import joblib
joblib.dump(model, 'rfclassifier.joblib')

