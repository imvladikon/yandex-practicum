# Data Scientist 

tasks and projects from the data science [course](https://practicum.yandex.com/profile/data-scientist/) by Yandex

### Final project
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imvladikon/ya-praktikum/blob/master/final/final_project.ipynb)

The telecom operator Interconnect would like to be able to forecast their churn of clients. If it's discovered that a user is planning to leave, they will be offered promotional codes and special plan options. Interconnect's marketing team has collected some of their clientele's personal data, including information about their plans and contracts.


### Sprint #16 - Unsupervised Learning

Theoretical parts and quizzes

### Sprint #15 - Computer Vision
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imvladikon/ya-praktikum/blob/master/sprint15/sprint15.ipynb)

Supermarket chain Good Seed is introducing a computer vision system for processing customer photos. Photofixation in the checkout area will help determine the age of customers in order to:
Analyze purchases and offer products that may interest buyers in particular age groups
Monitor clerks selling alcohol
Build a model that will determine the approximate age of a person from a photograph. To help, you'll have a set of photographs of people with their ages indicated.

### Sprint #14 - Machine Learning for Texts
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imvladikon/ya-praktikum/blob/master/sprint14/sprint14.ipynb)

The Film Junky Union, a new edgy community for classic movie enthusiasts, is developing a system for filtering and categorizing movie reviews. The goal is to train a model to automatically detect negative reviews. You'll be using a dataset of IMBD movie reviews with polarity labelling to build a model for classifying positive and negative reviews. It will need to reach an F1 score of at least 0.85.

### Sprint #13 - Time Series
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imvladikon/ya-praktikum/blob/master/sprint13/sprint13.ipynb)

Sweet Lift Taxi company has collected historical data on taxi orders at airports. To attract more drivers during peak hours, we need to predict the amount of taxi orders for the next hour. Build a model for such a prediction.
The RMSE metric on the test set should not be more than 48.

### Sprint #12 - Numerical Methods
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imvladikon/ya-praktikum/blob/master/sprint12/sprint12.ipynb)

Rusty Bargain used car sales service is developing an app to attract new customers. In that app, you can quickly find out the market value of your car. You have access to historical data: technical specifications, trim versions, and prices. You need to build the model to determine the value.
Rusty Bargain is interested in:
* the quality of the prediction
* the speed of the prediction
* the time required for training

### Sprint #11 - Linear Algebra
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imvladikon/ya-praktikum/blob/master/sprint11/sprint11.ipynb)

The Sure Tomorrow insurance company wants to protect its clients' data. Your task is to develop a data transforming algorithm that would make it hard to recover personal information from the transformed data. This is called data masking, or data obfuscation. You are also expected to prove that the algorithm works correctly. Additionally, the data should be protected in such a way that the quality of machine learning models doesn't suffer. You don't need to pick the best model. Follow these steps to develop a new algorithm:
* construct a theoretical proof using properties of models and the given task;
* formulate an algorithm for this proof;
* check that the algorithm is working correctly when applied to real data.

### Sprint #10 - Integrated project 2
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imvladikon/ya-praktikum/blob/master/sprint10/sprint10.ipynb)

Prepare a prototype of a machine learning model for Zyfra. The company develops efficiency solutions for heavy industry.
The model should predict the amount of gold recovered from gold ore. You have the data on extraction and purification.
The model will help to optimize the production and eliminate unprofitable parameters.
You need to:
* Prepare the data;
* Perform data analysis;
* Develop and train a model.
To complete the project, you may want to use documentation from pandas, matplotlib, and sklearn.
The next lesson is about the ore purification process. You will pick the information that is important for the model development.


### Sprint #9 - Machine Learning for Business
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imvladikon/ya-praktikum/blob/master/sprint09/sprint9.ipynb)

You work for the OilyGiant mining company. Your task is to find the best place for a new well.
Steps to choose the location:
Collect the oil well parameters in the selected region: oil quality and volume of reserves;
Build a model for predicting the volume of reserves in the new wells;
Pick the oil wells with the highest estimated values;
Pick the region with the highest total profit for the selected oil wells.
You have data on oil samples from three regions. Parameters of each oil well in the region are already known. Build a model that will help to pick the region with the highest profit margin. Analyze potential profit and risks using the Bootstrapping technique.


### Sprint #8 - Supervised learning
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imvladikon/ya-praktikum/blob/master/sprint08/sprint8.ipynb)

Beta Bank customers are leaving: little by little, chipping away every month. The bankers figured out it’s cheaper to save the existing customers rather than to attract new ones.
We need to predict whether a customer will leave the bank soon. You have the data on clients’ past behavior and termination of contracts with the bank.
Build a model with the maximum possible F1 score. To pass the project, you need an F1 score of at least 0.59. Check the F1 for the test set.
Additionally, measure the AUC-ROC metric and compare it with the F1.


### Sprint #7 - Introduction to machine learning
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imvladikon/ya-praktikum/blob/master/sprint07/sprint7.ipynb)

Mobile carrier Megaline has found out that many of their subscribers use legacy plans. They want to develop a model that would analyze subscribers' behavior and recommend one of Megaline's newer plans: Smart or Ultra.
You have access to behavior data about subscribers who have already switched to the new plans (from the project for the Statistical Data Analysis course). For this classification task, you need to develop a model that will pick the right plan. Since you’ve already performed the data preprocessing step, you can move straight to creating the model.
Develop a model with the highest possible accuracy. In this project, the threshold for accuracy is 0.75. Check the accuracy using the test dataset.


### Sprint #6 - Data Collection and Storage (SQL)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imvladikon/ya-praktikum/blob/master/sprint06/sprint6.ipynb)

You're working as an analyst for Zuber, a new ride-sharing company that's launching in Chicago. Your task is to find patterns in the available information. You want to understand passenger preferences and the impact of external factors on rides.
You'll study a database, analyze data from competitors, and test a hypothesis about the impact of weather on ride frequency.

### Sprint #5 - Integrated Project 1
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imvladikon/ya-praktikum/blob/master/sprint05/sprint5.ipynb)

You work for the online store Ice, which sells videogames all over the world. User and expert reviews, genres, platforms (e.g. Xbox or PlayStation), and historical data on game sales are available from open sources. You need to identify patterns that determine whether a game succeeds or not. This allows you to put your money on a potentially hot new item and plan advertising campaigns.
In front of you is data going back to 2016. Let’s imagine that it’s December 2016 and you’re planning a campaign for 2017.
The important thing is to get experience working with data. It doesn't really matter whether you're forecasting 2017 sales based on data from 2016 or 2027 sales based on data from 2026.
The data set contains the abbreviation ESRB (Entertainment Software Rating Board). The ESRB evaluates a game's content and assigns an appropriate age categories, such as Teen and Mature.

### Sprint #4 - Statistical data analysis
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imvladikon/ya-praktikum/blob/master/sprint04/sprint4.ipynb)

You work as an analyst for "Megaline", a state mobile operator. The company offers its clients two prepaid plans, Surf and Ultimate. The commercial department would like to know which of the plans is more profitable in order to adjust the advertising budget.
You are going to carry out a preliminary analysis of the plans based on a relatively small client selection. You'll have the data on 500 "Megaline" clients, specifically, who the clients are, where they are from, which plan they use, the number of calls made and SMS they sent in 2018. You have to analyse clients' behavior and work out the most profitable prepaid plan.

### Sprint #3 - Exploratory Data Analysis (EDA)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imvladikon/ya-praktikum/blob/master/sprint03/sprint3.ipynb)

You’ll have the data from Yandex.Realty provided for you. Working with data will not always be completely familiar - from time to time, you'll encounter data from peculiar sources, operating with peculiar measurements. Let's work with something exotic to keep you on your toes: a real estate agency has an archive of sales ads for realty in St. Petersburg, Russia, and the surrounding areas collected over the past few years. You’ll need to learn how to determine the market value of real estate properties. Your task is to define the parameters. This will make it possible to create an automated system that is capable of detecting anomalies and fraud.
There are two different types of data available for every apartment for sale. The first type is a user’s input. The second type is received automatically based upon the map data. This could be calculated, for example, based upon the distance from the downtown area, airport, the nearest park or body of water.


### Sprint #2 - Data Preprocessing
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imvladikon/ya-praktikum/blob/master/sprint02/sprint2.ipynb)

Your project is to prepare a report for a bank’s loan division. You’ll need to find out if a customer’s marital status and number of children has an impact on whether they will default on a loan. The bank already has some data on customers’ credit worthiness.
Your report will be considered when building a credit scoring of a potential customer. A credit scoring is used to evaluate the ability of a potential borrower to repay their loan.


### Sprint #1

Python and Data Analysis Basics, Introduction to Data Science



#### notes
for pre-commit hooks (converting jupyter notebooks to py/html files) need to install:

```bash
pip install pre-commit
pre-commit install
```

see [here](https://pre-commit.com/)
