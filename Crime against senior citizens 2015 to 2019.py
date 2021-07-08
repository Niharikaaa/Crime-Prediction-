#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[39]:


dataset1 = pd.read_csv(r"C:\Users\Niharika\OneDrive\Desktop\minor_project_csv_files\Crime_against_senior_citizen_2015-2017.csv")
dataset1.head()


# In[40]:


dataset2 = pd.read_csv(r"C:\Users\Niharika\OneDrive\Desktop\minor_project_csv_files\Crime_against_senior_citizen_2017-2019.csv")
dataset2.head()


# In[41]:


dataset = pd.DataFrame()
dataset['State/UT'] = dataset1['State/UT']
dataset['2015'] = dataset1['2015']
dataset['2016'] = dataset1['2016']
dataset['2017'] = dataset1['2017']
dataset['2018'] = dataset2['2018']
dataset['2019'] = dataset2['2019']
dataset


# In[42]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[43]:


dataset.iloc[:,0] = le.fit_transform(dataset.iloc[:,0])


# In[44]:


dataset


# In[45]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[46]:


X


# In[47]:


y


# In[48]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)


# ## Linear Regression

# In[49]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[50]:


y_pred = regressor.predict(X_test)


# In[51]:


y_pred


# In[52]:


pd.DataFrame(
{
    'ACTUAL': y_test,
    'PREDICTED': y_pred
}
)


# In[53]:


from sklearn import metrics


# In[54]:


metrics.mean_absolute_error(y_pred, y_test)


# In[55]:


metrics.r2_score(y_test, y_pred)


# ## Decision tree regression

# In[19]:


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)


# In[20]:


regressor.predict([[2,7,2,1,3]])


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


# # Support Vector Regression

# In[58]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'linear')
regressor.fit(X_train, y_train)


# In[59]:


y_pred = regressor.predict(X_test)


# In[60]:


y_pred


# In[61]:


pd.DataFrame(
{
    'ACTUAL': y_test,
    'PREDICTED': y_pred
}
)


# In[62]:


from sklearn import metrics


# In[63]:


metrics.mean_absolute_error(y_pred, y_test)


# In[64]:


metrics.r2_score(y_test, y_pred)


# In[65]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'poly')
regressor.fit(X_train, y_train)


# In[66]:


pd.DataFrame(
{
   'ACTUAL': y_test,
   'PREDICTED': y_pred
}
)


# In[67]:


from sklearn import metrics


# In[68]:


metrics.mean_absolute_error(y_pred, y_test)


# In[69]:


metrics.r2_score(y_test, y_pred)


# In[70]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'sigmoid')
regressor.fit(X_train, y_train)


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


# In[75]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)


# In[76]:


pd.DataFrame(
{
    'ACTUAL': y_test,
    'PREDICTED': y_pred
}
)


# In[77]:


from sklearn import metrics


# In[78]:


metrics.mean_absolute_error(y_pred, y_test)


# In[79]:


metrics.r2_score(y_test, y_pred)


# # Random Forest Regression

# In[80]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)


# In[81]:


y_pred = regressor.predict(X_test)


# In[82]:


y_pred


# In[83]:


pd.DataFrame(
{
    'ACTUAL': y_test,
    'PREDICTED': y_pred
}
)


# In[84]:


from sklearn import metrics


# In[85]:


metrics.mean_absolute_error(y_pred, y_test)


# In[86]:


metrics.r2_score(y_test, y_pred)


# In[87]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 5, random_state = 0)
regressor.fit(X_train, y_train)


# In[88]:


y_pred = regressor.predict(X_test)


# In[89]:


pd.DataFrame(
{
    'ACTUAL': y_test,
    'PREDICTED': y_pred
}
)


# In[90]:


from sklearn import metrics


# In[91]:


metrics.mean_absolute_error(y_pred, y_test)


# In[92]:


metrics.r2_score(y_test, y_pred)


# In[97]:


import numpy as np
import matplotlib.pyplot as plt
 
  
# creating the dataset
data = {'Linear_Regression':0.95, 'Decsion_Tree':14, 'SVR':0.98,
        'Random_Forest':0.61}
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

