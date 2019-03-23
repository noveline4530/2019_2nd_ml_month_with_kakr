
# coding: utf-8

# ## 03. 변수 만들기, 강력한 모델(xgbm, lgbm)

# * 캐글 대회에서 우승 모델로 많이 사용되어지는 xgbm과 lgbm에 대해 알아보자.
# * 그리고 여러개의 파생변수를 추가해 보자.

# In[30]:


import numpy as np   # linear algebra
import pandas as pd 


# In[31]:


df_train = pd.read_csv("../input/train.csv") # ..은 현재 위치의 위의 디렉터리를 가르킴.
df_test = pd.read_csv("../input/test.csv")
print(df_train.shape, df_test.shape)


# ### 간단한 시각화(상관계수 찍어보기)

# In[32]:


import seaborn as sns
import matplotlib.pyplot as plt

corr = df_train.corr()
plt.subplots(figsize=(15,10))
sns.heatmap(corr, annot=True, fmt=".3f")
plt.show()


# In[33]:


from sklearn.model_selection import train_test_split  # 데이터 나눠주기(알아서)


# In[34]:


y = df_train['price']  
X = df_train.drop('price', axis=1)  # 집가격을 뺀 변수컬럼을 X로 한다.
X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                            test_size=0.20, random_state=123)  


# In[35]:


# kernel에는 기본적으로 lightgbm, xgboost가 설치되어 있다.
import lightgbm as lgb
from xgboost import XGBRegressor


# In[36]:


X_train = X_train.drop('date', axis=1) 
X_valid = X_valid.drop('date', axis=1) 


# In[37]:


## lgbm
lightgbm = lgb.LGBMRegressor(random_state=123)
lightgbm.fit(X_train, y_train)
y_pred_lgbm = lightgbm.predict(X_valid)


# In[38]:


## xgbm
xgbm_model = XGBRegressor()
xgbm_model.fit(X_train, y_train)
y_pred_xgbm = xgbm_model.predict(X_valid)


# In[39]:


from sklearn.metrics import mean_squared_error        


# ### RMSE 구하기

# In[40]:


rmse = np.sqrt(mean_squared_error(y_pred_lgbm, y_valid))
print("RMSE(linear regression) : " + str(rmse))


# In[41]:


rmse = np.sqrt(mean_squared_error(y_pred_xgbm, y_valid))
print("RMSE(linear regression) : " + str(rmse))


# ## 실제 제출해보자(lgbm 이용)

# In[42]:


df_test.columns


# In[43]:


X_test = df_test.drop('date', axis=1)   # 실전 데이터 이용 
y_pred_lgbm = lightgbm.predict(X_test)  # 예측


# In[44]:


sub = pd.read_csv("../input/sample_submission.csv")
sub['price'] = y_pred_lgbm
sub.to_csv("mySolution_lgbm.csv", index=False)


# ### 교차 검증
