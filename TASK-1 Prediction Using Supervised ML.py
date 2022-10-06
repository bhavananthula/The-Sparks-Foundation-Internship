#!/usr/bin/env python
# coding: utf-8

# # GRIP-The Sparks Foundation
# 

# # Data Science and business Analytics
# 

# # Task-1 :- Prediction using supervised ML
# 

# # Task Goal - To predict percentage of an student based on the no. of study hours
# 

# # Name:Bhavana Ananthula

# **STEP 1:Importing Libraries**

# In[1]:


#importing important libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#reading the data set
data = pd.read_csv('http://bit.ly/w-data')


# **STEP 2:Analysing Dataset**

# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.info() ##checking for null values


# In[6]:


data.describe()


# **STEP 3:Visualizing Data**

# In[7]:


data.columns #Visualizing the data


# In[8]:


sns.pairplot(data)


# In[9]:


sns.heatmap(data.corr(), annot=True)


# In[10]:


data.corr()


# **STEP 4:Training the Data**

# In[11]:


x = np.asanyarray(data['Hours'])
y = np.asanyarray(data['Scores'])


# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=0)


# In[13]:


lr = LinearRegression()


# In[14]:


lr.fit(np.array(x_train).reshape(-1,1), np.array(y_train).reshape(-1,1))


# In[15]:


print('Coefficients: ', lr.coef_)
print("Intercept: ", lr.intercept_)


# **STEP 5:Comparing Data(Actual vs Predicted)**

# In[16]:


#plotting the traing set
data.plot(kind='scatter', x='Hours', y='Scores', figsize=(10,5), color='red')
plt.plot(x_train, lr.coef_[0]*x_train + lr.intercept_, color='blue')
plt.title('Training Set')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.show()


# In[17]:


predict=lr.predict(np.array(x_test).reshape(-1,1))
predict


# In[18]:


df = pd.DataFrame(np.c_[x_test,y_test,predict], columns=['Hours', 'Actual Score', 'Predicted Scores'])
df


# In[19]:


#testing with given data

hours= [9.25]
pred=lr.predict([hours])
print("NO. of Hours = {}".format(hours))
print("Predicted scores = {}".format(pred[0]))


# In[20]:


from sklearn import metrics
print("Mean Absolute error:", metrics.mean_absolute_error(y_test,predict))
print("Mean Squared error:", metrics.mean_squared_error(y_test,predict))


# # CONCLUSION:

# **If the student studies 9.25 hours per day,the score may be 92.91505723**

# In[ ]:




