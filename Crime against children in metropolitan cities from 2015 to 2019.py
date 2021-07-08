#!/usr/bin/env python
# coding: utf-8

# # Crime against children in metropolitican cities from 2015 to 2019

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn


# In[4]:


dataset1 = pd.read_csv(r"C:\Users\Niharika\OneDrive\Desktop\minor_project_csv_files\Crime_Against_Children_in_Metropolitan_Cities_ 2015_2017.csv")
dataset1


# In[5]:


dataset2 = pd.read_csv(r"C:\Users\Niharika\OneDrive\Desktop\minor_project_csv_files\Crime_Against_Children_in_Metropolitan_Cities_ 2016_2018.csv")
dataset2


# In[7]:


dataset3 = pd.read_csv(r"C:\Users\Niharika\OneDrive\Desktop\minor_project_csv_files\Crime_Against_Childre_in_Metropolitan_ Cities _2017_2019.csv")
dataset3


# In[8]:


dataset = pd.DataFrame()
dataset['City'] = dataset1['City']
dataset['2015'] = dataset1['2015']
dataset['2016'] = dataset1['2016']
dataset['2017'] = dataset1['2017']
dataset['2018'] = dataset2['2018']
dataset['2019'] = dataset3['2019']
dataset


# In[9]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[10]:


dataset.iloc[:,0] = le.fit_transform(dataset.iloc[:,0])


# In[11]:


dataset


# In[12]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[13]:


X


# In[14]:


y


# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[16]:


X_train


# In[17]:


y_train


# In[18]:


X_test


# In[19]:


y_test


# # Linear Regression

# In[20]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[21]:


y_pred = regressor.predict(X_test)


# In[22]:


y_pred


# In[23]:


pd.DataFrame(
{
    'ACTUAL': y_test,
    'PREDICTED': y_pred
}
)


# In[24]:


from sklearn import metrics


# In[25]:


metrics.mean_absolute_error(y_pred, y_test)


# In[26]:


metrics.r2_score(y_test, y_pred)


# # Support Vector Regression (SVR)

# In[27]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'linear')
regressor.fit(X_train, y_train)


# In[28]:


y_pred = regressor.predict(X_test)


# In[29]:


y_pred


# In[30]:


pd.DataFrame(
{
    'ACTUAL': y_test,
    'PREDICTED': y_pred
}
)


# In[31]:


from sklearn import metrics


# In[32]:


metrics.mean_absolute_error(y_pred, y_test)


# In[33]:


metrics.r2_score(y_test, y_pred)


# In[34]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'poly')
regressor.fit(X_train, y_train)


# In[35]:


y_pred = regressor.predict(X_test)


# In[36]:


y_pred


# In[37]:


pd.DataFrame(
{
    'ACTUAL': y_test,
    'PREDICTED': y_pred
}
)


# In[38]:


from sklearn import metrics


# In[39]:


metrics.mean_absolute_error(y_pred, y_test)


# In[40]:


metrics.r2_score(y_test, y_pred)


# In[41]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'sigmoid')
regressor.fit(X_train, y_train)


# In[42]:


y_pred = regressor.predict(X_test)


# In[43]:


pd.DataFrame(
{
    'ACTUAL': y_test,
    'PREDICTED': y_pred
}
)


# In[44]:


from sklearn import metrics


# In[45]:


metrics.mean_absolute_error(y_pred, y_test)


# In[46]:


metrics.r2_score(y_test, y_pred)


# In[47]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)


# In[48]:


y_pred = regressor.predict(X_test)


# In[49]:


y_pred


# In[50]:


pd.DataFrame(
{
    'ACTUAL': y_test,
    'PREDICTED': y_pred
}
)


# In[51]:


from sklearn import metrics


# In[52]:


metrics.mean_absolute_error(y_pred, y_test)


# In[53]:


metrics.r2_score(y_test, y_pred)


# # Decision Tree Regression

# In[54]:


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)


# In[55]:


y_pred = regressor.predict(X_test)


# In[56]:


y_pred


# In[57]:


pd.DataFrame(
{
    'ACTUAL': y_test,
    'PREDICTED': y_pred
}
)


# In[58]:


from sklearn import metrics


# In[59]:


metrics.mean_absolute_error(y_pred, y_test)


# In[60]:


metrics.r2_score(y_test, y_pred)


# # Random Forest Regression# 

# In[61]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)


# In[62]:


y_pred = regressor.predict(X_test)


# In[63]:


y_pred


# In[64]:


pd.DataFrame(
{
    'ACTUAL': y_test,
    'PREDICTED': y_pred
}
)


# In[65]:


from sklearn import metrics


# In[66]:


metrics.mean_absolute_error(y_pred, y_test)


# In[67]:


metrics.r2_score(y_test, y_pred)


# In[68]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 5, random_state = 0)
regressor.fit(X_train, y_train)


# In[69]:


y_pred = regressor.predict(X_test)


# In[70]:


y_pred


# In[71]:


pd.DataFrame(
{
    'ACTUAL': y_test,
    'PREDICTED': y_pred
}
)


# In[72]:


from sklearn import metrics


# In[73]:


metrics.mean_absolute_error(y_pred, y_test)


# In[74]:


metrics.r2_score(y_test, y_pred)


# In[76]:


import numpy as np
import matplotlib.pyplot as plt
 
  
# creating the dataset
data = {'Linear_Regression': 0.97, 'Decsion_Tree':0.06, 'SVR':0.97,
        'Random_Forest':0.09}
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

