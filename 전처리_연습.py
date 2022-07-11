#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

csv_file_path = os.getenv('HOME')+'/aiffel/data_preprocess/data/vgsales.csv'
vgsales = pd.read_csv(csv_file_path) 
vgsales.head()


# ## 결측치 제거

# In[5]:


vgsales.info()


# In[ ]:





# In[37]:


vgsales.isnull().sum()


# In[9]:


vgsales.describe()


# In[27]:


print((vgsales['Publisher'] == 'Nintendo').count())


# In[25]:


vgsales['Publisher'] == 'Nintendo'


# In[34]:


vgsales.dropna(inplace=True)


# In[36]:


vgsales.info()


# In[55]:


vgsales.describe(include='all')


# In[62]:


vgsales.isnull().sum()


# - 결측치는 Year과 Publisher에 각각 271 ,51개로 전체 컬럼 갯수 16598에 비하면 매우 적은 양으로 데이터를 제거해도 분석에 지장이 없어 보인다. 
# - 다른 값으로 대체할려고 해보니 Publisher의 경우는 다양한 값들이 포진해있다. 이거를 간단하게 확인하는 방법이 있을 건데 나는 노가다했다. 일단 천천히 찾아보기로 하고 넘어가자
# - 갯수가 미약하다는것이 제일 큰 제거 요인이다.

# ## 중복데이터 제거

# In[38]:


vgsales.duplicated()


# In[42]:


vgsales[vgsales.duplicated()]


# - 중복된 데이터는 없는 것으로 확인

# ## 이상치 확인하기

# In[51]:


def outlier(df, col, z):
    return df[abs(df[col] - np.mean(df[col]))/np.std(df[col])>z].index


# In[56]:


vgsales.loc[outlier(vgsales, 'JP_Sales', 2)]
    


# - 이상치도 없을 것으로 추측

# ## 원-핫 인코딩

# In[60]:


country = pd.get_dummies(vgsales['Name'])


# In[61]:


country.head()


# In[ ]:





# ## 정규화_MinMax

# In[57]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[58]:


scaler.fit_transform(vgsales)


# In[ ]:




