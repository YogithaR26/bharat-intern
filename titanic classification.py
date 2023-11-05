#!/usr/bin/env python
# coding: utf-8

# ## BHARAT INTERNSHIP
# 
#   ## NAME-YOGITHA RAJULAPATI
#   
#   ## TASK 2-TITANIC CLASSIFICATION
#   - In this we predicts if a passenger will survive on the titanic or not
#   
#   # Import Libraries

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

# # Loading the dataset

# In[3]:


titanic=pd.read_csv(r"C:\Users\smily\Downloads\archive\tested.csv")


# In[4]:


titanic


# # Cleaning the dataset

# In[4]:


titanic.shape


# In[5]:


titanic.info()     #datatype info


# In[6]:


titanic.describe()        #statistical info


# In[7]:


titanic.head()


# In[8]:


titanic.columns


# # Exploratory Data Analysis

# In[9]:


titanic["Sex"].value_counts


# In[10]:


titanic['Survived'].value_counts()       #count of no.of survivors


# # Countplot of survived vs not survived

# In[11]:


sns.countplot(x='Survived',data=titanic)


# # Male vs Female Survival

# In[12]:


sns.countplot(x='Survived',data=titanic,hue='Sex')


# In[13]:


sns.countplot(x='Survived',hue='Pclass',data=titanic)


# # Missing Data

# In[14]:


titanic.isna()


# In[15]:


titanic.isnull().sum()


# # Visualize null values

# In[16]:


sns.heatmap(titanic.isna())


# In[17]:


(titanic['Age'].isna().sum()/len(titanic['Age']))*100


# In[18]:


(titanic['Cabin'].isna().sum()/len(titanic['Cabin']))*100


# In[19]:


sns.distplot(titanic['Age'].dropna(),kde=False,color='blue',bins=40)


# In[20]:


titanic['Age'].hist(bins=40,color='blue',alpha=0.4)


# In[21]:


sns.countplot(x='SibSp',data=titanic)


# In[22]:


titanic['Fare'].hist(color='red',bins=40,figsize=(8,4))


# In[23]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=titanic,palette='winter')


# In[24]:


titanic['Age'].fillna(titanic['Age'].mean(),inplace=True)


# In[25]:


titanic['Age'].isna().sum()


# In[26]:


sns.heatmap(titanic.isna())


#  - We can see cabin column has a number of null values, as such we can not use it for prediction, Hence we will drop it

# In[27]:


titanic.drop('Cabin',axis=1,inplace=True)


# In[28]:


titanic.head()


# # Preparing Data for Model

# In[29]:


titanic.info()


# # convert sex column to numerical values

# In[30]:


gender=pd.get_dummies(titanic['Sex'],drop_first=True)


# In[31]:


titanic['Gender']=gender


# In[2]:


titanic.head()


# # Drop the columns which are not required

# In[1]:





# In[33]:


x=titanic[['PassengerId','Pclass','Age','SibSp','Parch','Fare','Gender']]
y=titanic['Survived']


# In[34]:


y


# In[72]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.22,random_state=41)


# In[73]:


yr=LogisticRegression()


# In[74]:


yr.fit(x_train,y_train)


# In[76]:


predict=yr.predict(x_train)           #predict


# In[78]:


pd.DataFrame(confusion_matrix(y_train,predict),columns=['Predicted No','Predicted Yes'],index=['Actual No','Actual Yes'])


# In[79]:


print(classification_report(y_train,predict))


# In[ ]:




