#!/usr/bin/env python
# coding: utf-8

# The Sure Tomorrow insurance company wants to protect its clients' data. Your task is to develop a data transforming algorithm that would make it hard to recover personal information from the transformed data. Prove that the algorithm works correctly
# 
# The data should be protected in such a way that the quality of machine learning models doesn't suffer. You don't need to pick the best model.

# ## 1. Data downloading

# In[ ]:





# ## 2. Multiplication of matrices

# In this task, you can write formulas in *Jupyter Notebook.*
# 
# To write the formula in-between the text, frame it with dollar signs \\$; if it should be outside the text —  with double signs \\$\\$. These formulas are written in markup language *LaTeX.* 
# 
# For example, we wrote down linear regression formulas. You can copy and edit them to solve the task.
# 
# You don't have to use *LaTeX*.

# Denote:
# 
# - $X$ — feature matrix (zero column consists of unities)
# 
# - $y$ — target vector
# 
# - $P$ — matrix by which the features are multiplied
# 
# - $w$ — linear regression weight vector (zero element is equal to the shift)

# Predictions:
# 
# $$
# a = Xw
# $$
# 
# Training objective:
# 
# $$
# \min_w d_2(Xw, y)
# $$
# 
# Training formula:
# 
# $$
# w = (X^T X)^{-1} X^T y
# $$

# ** Answer:** ...
# 
# ** Justification:** ...

# ## 3. Transformation algorithm

# ** Algorithm**
# 
# ...

# ** Justification**
# 
# ...

# ## 4. Algorithm test

# In[ ]:





# ## Checklist

# Type 'x' to check. Then press Shift+Enter.

# - [x]  Jupyter Notebook is open
# - [ ]  Code is error free
# - [ ]  The cells with the code have been arranged in order of execution
# - [ ]  Step 1 performed: the data was downloaded
# - [ ]  Step 2 performed: the answer to the matrix multiplication problem was provided
#     - [ ]  The correct answer was chosen
#     - [ ]  The choice was justified
# - [ ]  Step 3 performed: the transform algorithm was proposed
#     - [ ]  The algorithm was described
#     - [ ]  The algorithm was justified
# - [ ]  Step 4 performed: the algorithm was tested
#     - [ ]  The algorithm was realized
#     - [ ]  Model quality was assessed before and after the transformation

# In[ ]:




