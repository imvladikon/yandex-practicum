#!/usr/bin/env python
# coding: utf-8

# Hello, my name is Artem. I'm going to review your project!
# 
# You can find my comments in <font color='green'>green</font>, <font color='blue'>blue</font> or <font color='red'>red</font> boxes like this:
# 
# <div class="alert alert-block alert-success">
# <b>Success:</b> if everything is done succesfully
# </div>
# 
# <div class="alert alert-block alert-info">
# <b>Improve: </b> "Improve" comments mean that there are tiny corrections that could help you to make your project better.
# </div>
# 
# <div class="alert alert-block alert-danger">
# <b>Needs fixing:</b> if the block requires some corrections. Work can't be accepted with the red comments.
# </div>
# 
# ### <font color='orange'>General feedback</font>
# * Thank you for submitting your project! I am really impressed with it. 
# * Glad to see that the notebook is structured. Keep it up!
# * It was a pleasure to review your project.
# * You' achieved the required score. Congratulations!
# * I've found some tiny mistakes in your project. They'll be easy to fix.
# * You can also make your project even better if you work on the "improve" comments.
# * While there's room for improvement, on the whole, your project is looking good.
# * I believe you can easily fix it!

# >Thank you!

# ### <font color='orange'>General feedback</font>
# * I really appreciate the corrections you sent in! Thanks for taking the time to do so.
# * horizontal_flip should be added as a parameter in ImageDataGenerator. Adding it to `train_datagen.flow_from_dataframe` does not make sense.
# * In general, all errors were fixed& Thank you.
# * This project is accepted. Good luck in the future!

# # 1. Exploratory data analysis

# In[43]:


import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np


# In[2]:


labels = pd.read_csv('/datasets/faces/labels.csv')
train_datagen = ImageDataGenerator(rescale=1./255)
train_gen_flow = train_datagen.flow_from_dataframe(
        dataframe=labels,
        directory='/datasets/faces/final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        seed=12345)


# In[3]:


labels.info()


# <div class="alert alert-block alert-success">
# <b>Success:</b> Data loading was done OK.
# </div>

# In[33]:


sns.distplot(labels["real_age"],  hist = True, bins=30)


# In[34]:


sns.boxplot(labels["real_age"])


# a bit skewed normal distribution

# In[6]:


features, target = next(train_gen_flow)


# In[35]:


features.shape


# In[17]:


for age, photo in zip(target[:10], features[:10]):
    plt.figure()
    plt.title(age)
    plt.imshow(photo)


# In[49]:


photo.shape


# we have rotated images (and therefore have to add rotation augmentated images in a training process)
# 224x224 size, and also we have grayscaled images

# <div class="alert alert-block alert-info">
# <b>Improve: </b> It would be better if the age of the person was set in the title.
# </div>

# <div class="alert alert-block alert-danger">
# <b>Needs fixing:</b> Please provide some findings and comments about target feature distribution and images (rotation, color, size, etc).
# </div>

# >sure, added

# # 2. Model training

# Transfer the model training code and the result of printing on the screen here.
# 
# 
# (The code in this section is run on a separate GPU platform, so it is not designed as a cell with a code, but as a code in a text cell)

# ```python
# 
# import numpy as np
# import pandas as pd
# from tensorflow.keras.datasets import fashion_mnist
# from tensorflow.keras.layers import Conv2D, Flatten, AvgPool2D, Dense, GlobalAveragePooling2D
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications.resnet import ResNet50 
# 
# 
# def load_train(path):
#     labels = pd.read_csv(path+'labels.csv')                                                     
#     train_datagen = ImageDataGenerator(rescale= 1./255, validation_split=0.25)  
#     train_datagen_flow = train_datagen.flow_from_dataframe(
#         dataframe = labels,
#         directory = path + 'final_files/',
#         x_col='file_name',
#         y_col='real_age',
#         target_size=(224, 224),
#         batch_size=32,
#         class_mode='raw',
#         subset='training',
#         horizontal_flip=True,
#         seed=42)
#     return train_datagen_flow
# 
# def load_test(path):
#     labels = pd.read_csv(path+'labels.csv')  
#     test_datagen = ImageDataGenerator(rescale= 1./255, validation_split=0.25)  
#     test_datagen_flow = test_datagen.flow_from_dataframe(
#         dataframe = labels,
#         directory = path + 'final_files/',
#         x_col='file_name',
#         y_col='real_age',
#         target_size=(224, 224),
#         batch_size=32,
#         class_mode='raw',
#         subset='validation', 
#         seed=42)
#     
#     return test_datagen_flow
# 
# def create_model(input_shape):
#     
#     backbone = ResNet50(input_shape= input_shape,
#                     weights='imagenet', 
#                     include_top= False)
#                      
#     model = Sequential()
#     model.add(backbone)
#     model.add(GlobalAveragePooling2D())
#     model.add(Dense(1, activation='relu'))
#     optimizer = Adam(lr=0.0005)
#     model.compile(optimizer=optimizer, loss='mean_squared_error',metrics=['mae']) 
#     return model
#     
#     
# def train_model(model, train_data, test_data, batch_size=None, epochs=10,
#                 steps_per_epoch=None, validation_steps=None):
#     
#     if steps_per_epoch is None:
#         steps_per_epoch = len(train_data)
#     if validation_steps is None:
#         validation_steps = len(test_data)
#     model.fit(train_data, 
#               validation_data= test_data,
#               batch_size=batch_size, epochs=epochs,
#               steps_per_epoch=steps_per_epoch,
#               validation_steps=validation_steps,
#               verbose=2)
#     return model
# 
# ```

# <div class="alert alert-block alert-info">
# <b>Improve: </b> Using horizontal flip at the training part could help you to increase the training set with valid images.
# </div>

# >right, thank you, added.

# ```
# 
# Epoch 1/10
# 2020-10-26 22:46:50.462891: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
# 2020-10-26 22:46:50.988793: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 178/178 - 99s - loss: 205.0102 - mae: 10.5819 - val_loss: 317.3405 - val_mae: 13.2126
# Epoch 2/10
# 178/178 - 41s - loss: 96.4598 - mae: 7.4412 - val_loss: 443.4091 - val_mae: 15.8325
# Epoch 3/10
# 178/178 - 41s - loss: 64.0527 - mae: 6.1343 - val_loss: 301.9018 - val_mae: 12.8919
# Epoch 4/10
# 178/178 - 41s - loss: 43.2806 - mae: 5.0341 - val_loss: 149.8389 - val_mae: 9.1190
# Epoch 5/10
# 178/178 - 41s - loss: 29.6981 - mae: 4.2136 - val_loss: 104.9151 - val_mae: 7.5741
# Epoch 6/10
# 178/178 - 41s - loss: 20.6762 - mae: 3.5109 - val_loss: 81.3642 - val_mae: 6.6892
# Epoch 7/10
# 178/178 - 40s - loss: 17.3920 - mae: 3.2108 - val_loss: 87.1265 - val_mae: 6.9833
# Epoch 8/10
# 178/178 - 41s - loss: 13.5037 - mae: 2.8095 - val_loss: 73.9750 - val_mae: 6.4053
# Epoch 9/10
# 178/178 - 41s - loss: 10.7918 - mae: 2.5193 - val_loss: 80.2930 - val_mae: 6.7702
# Epoch 10/10
# 178/178 - 40s - loss: 8.8153 - mae: 2.2726 - val_loss: 69.1264 - val_mae: 6.2419
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to  
#   ['...']
# 60/60 - 10s - loss: 69.1264 - mae: 6.2419
# Test MAE: 6.2419
# 
# ```

# <div class="alert alert-block alert-success">
# <b>Success:</b> You've achieved a great score! Well done!
# </div>

# # 3. Trained model analysis

# Model reached demanded value metric, but as we see for getting MAE less than 7, we also could use earlystopping and get result earlier. Also good practice is plotting learning curve and checking 

# <div class="alert alert-block alert-success">
# <b>Success:</b> Great analysis.
# </div>

# <div class="alert alert-block alert-info">
# <b>Improve: </b> You could say some words about parameters that you've used. Some words about optimzer and learning rate?
# </div>

# >I decided choose half from default value of the learning rate 0.0005 and Adam optimizers which is well balanced on a perfomance and learning sides, it's better firstly try Adam

# # Checklist

# - [x]  Notebook was opened
# - [x]  The code is error free
# - [x]  The cells with code have been arranged by order of execution
# - [x]  The exploratory data analysis has been performed
# - [x]  The results of the exploratory data analysis have been transferred to the final notebook
# - [x]  The model's MAE score is not higher than 8
# - [x]  The model training code has been copied to the final notebook
# - [x]  The model's printing on the screen result has been transferred to the final notebook
# - [x]  The findings have been provided based on the results of the model training
