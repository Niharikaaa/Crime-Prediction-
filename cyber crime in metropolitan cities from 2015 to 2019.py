#!/usr/bin/env python
# coding: utf-8

# # CYBER CRIME IN METROPOLITAN CITIES FROM 2015 TO 2019

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn


# In[2]:


dataset1 = pd.read_csv(r"C:\Users\Niharika\OneDrive\Desktop\minor_project_csv_files\Cyber_Crimes_in_Metropolitan_Cities-2015-2017.csv")
dataset1


# In[3]:


dataset2 = pd.read_csv(r"C:\Users\Niharika\OneDrive\Desktop\minor_project_csv_files\Cyber_Crimes_in_Metropolitan_Cities-2016-2018.csv")
dataset2


# In[4]:


dataset3 = pd.read_csv(r"C:\Users\Niharika\OneDrive\Desktop\minor_project_csv_files\Cyber_Crimes_in_Metropolitan_Cities-2017-2019.csv")
dataset3


# In[5]:


dataset = pd.DataFrame()
dataset['City'] = dataset1['City']
dataset['2015'] = dataset1['2015']
dataset['2016'] = dataset1['2016']
dataset['2017'] = dataset1['2017']
dataset['2018'] = dataset2['2018']
dataset['2019'] = dataset3['2019']
dataset


# In[6]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[7]:


dataset.iloc[:,0] = le.fit_transform(dataset.iloc[:,0])


# In[8]:


dataset


# In[9]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[10]:


X


# In[11]:


y


# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[13]:


X_train


# In[14]:


y_train


# In[15]:


X_test


# In[16]:


y_test


# ## Linear Regression

# In[17]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[18]:


y_pred = regressor.predict(X_test)


# In[19]:


y_pred


# In[20]:


pd.DataFrame(
{
    'ACTUAL': y_test,
    'PREDICTED': y_pred
}
)


# In[21]:


from sklearn import metrics


# In[22]:


metrics.mean_absolute_error(y_pred, y_test)


# In[23]:


metrics.r2_score(y_test, y_pred)


# # Support Vector Regression (SVR)

# In[24]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'linear')
regressor.fit(X_train, y_train)


# In[25]:


y_pred = regressor.predict(X_test)


# In[26]:


y_pred


# In[27]:


pd.DataFrame(
{
    'ACTUAL': y_test,
    'PREDICTED': y_pred
}
)


# In[28]:


from sklearn import metrics


# In[29]:


metrics.mean_absolute_error(y_pred, y_test)


# In[30]:


metrics.r2_score(y_test, y_pred)


# In[46]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'poly')
regressor.fit(X_train, y_train)


# In[47]:


y_pred = regressor.predict(X_test)


# In[48]:


y_pred


# In[49]:


pd.DataFrame(
{
    'ACTUAL': y_test,
    'PREDICTED': y_pred
}
)


# In[50]:


from sklearn import metrics


# In[51]:


metrics.mean_absolute_error(y_pred, y_test)


# In[52]:


metrics.r2_score(y_test, y_pred)


# In[53]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'sigmoid')
regressor.fit(X_train, y_train)


# In[54]:


y_pred = regressor.predict(X_test)


# In[55]:


pd.DataFrame(
{
    'ACTUAL': y_test,
    'PREDICTED': y_pred
}
)


# In[56]:


from sklearn import metrics


# In[57]:


metrics.mean_absolute_error(y_pred, y_test)


# In[58]:


metrics.r2_score(y_test, y_pred)


# # Decision Tree Regression

# In[31]:


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)


# In[32]:


y_pred = regressor.predict(X_test)


# In[33]:


y_pred


# In[34]:


pd.DataFrame(
{
    'ACTUAL': y_test,
    'PREDICTED': y_pred
}
)


# In[35]:


from sklearn import metrics


# In[36]:


metrics.mean_absolute_error(y_pred, y_test)


# In[37]:


metrics.r2_score(y_test, y_pred)


# # Random Forest Regression
# 

# In[38]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)


# In[39]:


y_pred = regressor.predict(X_test)


# In[40]:


y_pred


# In[41]:


pd.DataFrame(
{
    'ACTUAL': y_test,
    'PREDICTED': y_pred
}
)


# In[42]:


from sklearn import metrics


# In[43]:


metrics.mean_absolute_error(y_pred, y_test)


# In[44]:


metrics.r2_score(y_test, y_pred)


# In[45]:


import numpy as np
import matplotlib.pyplot as plt
 
  
# creating the dataset
data = {'Linear_Regression': -87.80, 'Decsion_Tree':-0.96, 'SVR':-0.94,
        'Random_Forest':-13.55}
algorithms = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(algorithms, values, color ='maroon',
        width = 0.4)
 
plt.xlabel("r2 score of various algorithms")
plt.ylabel("Value of r2 score")
plt.title("R2 Score of various algorithms")
plt.show()

