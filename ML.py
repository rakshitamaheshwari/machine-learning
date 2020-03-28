
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


dataset=pd.read_csv('/root/Desktop/Salary_Data.csv')


# In[4]:


dataset.head()


# In[5]:


y=dataset.iloc[:,1] #for 2d array


# In[6]:


x=dataset.iloc[:,0:1]


# In[7]:


from sklearn.linear_model import LinearRegression


# In[8]:


model=LinearRegression()


# In[9]:


model.fit(x,y) #model trained


# In[11]:


model.coef_#formula coeff.  for y=cx


# In[12]:


model.intercept_


# In[13]:


model.predict([[5]])


# In[14]:


import matplotlib.pyplot as plt


# In[15]:


plt.scatter(x,y)

