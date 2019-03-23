
# coding: utf-8

# ## 02. 데이터 나누기, 평가하기

# * 캐글은 많게는 10번 적게는 5번 밖에 제출하지 못한다.

# ### 그러면 어떻게 할까?
# * 학습용 데이터를 나눠서 모델을 만들고 평가해 보자. 그리고 가장 좋은 친구로 골라 제출해보기
# 

# ### 머신러닝 모델에 적용시킬 학습용 문제를 두 가지로 나누기
# * 학습용 문제(df_train)
# * 학습용 문제와(train) 
# * 실전 모의고사(valid)를 분리시켜보자는 것이다.
# * 그리고 실전 문제(df_test)

# In[1]:


import numpy as np   # linear algebra
import pandas as pd 


# In[2]:


df_train = pd.read_csv("../input/train.csv") # ..은 현재 위치의 위의 디렉터리를 가르킴.
df_test = pd.read_csv("../input/test.csv")
print(df_train.shape, df_test.shape)


# ### 간단한 시각화(상관계수 찍어보기)

# In[39]:


import seaborn as sns
import matplotlib.pyplot as plt

corr = df_train.corr()
plt.subplots(figsize=(15,10))
sns.heatmap(corr, annot=True, fmt=".3f")
plt.show()


# In[4]:


from sklearn import linear_model   # 모델 만들기 
from sklearn.model_selection import train_test_split  # 데이터 나눠주기(알아서)


# In[5]:


sel = ['sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms']


# In[6]:


y = df_train['price']  # 예측하려고 하는 값(집가격)
X = df_train[sel]      # 예측할 때, 모델을 학습시킬 때 사용하는 데이터
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=123)  


# In[9]:


print(X_train.shape)   # 학습용 문제
print(X_valid.shape)   # 평가용 문제(실전 모의고사)
print(y_train.shape)   # 학습용 답
print(y_valid.shape)   # 평가용 답(실전 모의고사)


# ### 이해해 보기
# * 01. 우리는 학습용 문제를 이용해서 모델을 학습시킨다.
# * 02. 공부시킨 후, 똑똑해진 모델을 이용해서 평가용 문제를 풀게 한다.
# * 03. 문제를 풀면 **예측한 답이 나오고(평가용 답) 실제 답과 비교**해서 어느정도 차이가 있는지 평가할 수 있다.

# In[11]:


model = linear_model.LinearRegression()  # 사용할 모델 선택 
model.fit(X_train, y_train)              # 학습용 문제와 답을 이용하여 공부(학습)시키기
y_pred_linear = model.predict(X_valid)   # 학습한 모델(똑똑해진?)친구로 실전모의고사 데이터로 집값 예측


# In[12]:


from sklearn.metrics import mean_squared_error        # 실제값과 예측값의 차이에 대한 평가 지표를 계산해줌.


# ### MSE 구하기

# In[13]:


print("Mean Squared Error(linear regression) : " + str(mean_squared_error(y_pred_linear, y_valid)))


# In[14]:


np.sqrt(100)


# ### RMSE 구하기

# In[15]:


rmse = np.sqrt(mean_squared_error(y_pred_linear, y_valid))
print("MSE(linear regression) : " + str(rmse))


# ### Knn 모델 구해보기

# In[34]:


from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=5)
neigh.fit(X_train, y_train)


# In[35]:


y_pred_knn = neigh.predict(X_valid)
y_pred_knn.shape


# In[36]:


rmse = np.sqrt(mean_squared_error(y_pred_knn, y_valid))
print("MSE(linear regression) : " + str(rmse))


# ## REF : 참고
# * scikit 관련 : https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors
# * knn 모델 : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor
