
# coding: utf-8

# ## 나의 첫 모델 만들기

# ### 캐글 대회 참여 방법

# (1) 대회 참여 URL : https://www.kaggle.com/c/2019-2nd-ml-month-with-kakr 접속 <br>
# (2) 참여방법 : 다음 링크를 눌러 대회 페이지에 다시 들어오셔야 참여 가능합니다. 무분별한 외국인들의 참가를 방지하기 위함입니다.  https://bit.ly/2UuQvtU  (링크 선택) <br>
# (3) 코드 작성 여러가지 방법 - Kaggle Kernel, 구글 Colaboratory, 주피터 노트북, Spyder 등. <br>
# (4) 이번 사용 방법 - Kaggle Kernel 이용

# ## 01. 데이터 파일 확인 및 불러오기

# In[4]:


import numpy as np   # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[7]:


df_train = pd.read_csv("../input/train.csv") # ..은 현재 위치의 위의 디렉터리를 가르킴.
df_test = pd.read_csv("../input/test.csv")
print(df_train.shape, df_test.shape)


# ## 02. 어떤 변수(컬럼)이 있을까?

# In[8]:


print(df_train.columns)
print(df_test.columns)


# ## 03. 예측 모델을 만들어, 이 모델을 이용하여 예측한 값을 제출해 본다.
# * 01. 사용할 모델 선택
# * 02. 모델에 학습를 시킨다.(학습시킬 문제와 답이 필요)
# * 03. 학습를 시킨 모델로 예측을 한다.

# ### 03-01 학습시킬 실전 문제(데이터)와 답(예측할 값)을 준비한다.

# * 예측할 값은 price이고,
# * price를 예측할 때, 사용하는 변수(컬럼)은 몇개를 쓸지 우리가 지정할 수 있다.
# * 집의 가격에 미치는 건 역시 집의 크기가 아닐까? 그리고 방의 개수 관련 있는 것을 가지고 한번 모델을 만들어보자.

# In[9]:


sel = ['sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms']
sel


# In[22]:


train_problem = df_train[sel]   # 전체 데이터 중의 4개의 컬럼(변수)를 일부 선택
solution = df_train['price']
real_problem = df_test[sel]
print(train_problem.shape)   # 학습시킬 문제수
print(solution.shape)        # 학습시킬 정답
print(real_problem.shape)    # 실전문제 문제수 


# ### 03-02 모델을 선택(학습방법 선택), 학습를 시킨다. 그리고 학습을 시키고 예측을 해보기

# In[23]:


from sklearn import linear_model                 # 선형회귀라는 학습 모델을 선택한다.
from sklearn.metrics import mean_squared_error   # 예측 했을 때, 평가해주는 도구


# In[24]:


model = linear_model.LinearRegression()
model


# In[25]:


model.fit(train_problem, solution)


# In[27]:


real_solution = model.predict(real_problem)  
real_solution


# ### 03-03 답안지을 받아서 실제 마지막으로 제출해 보자.

# In[28]:


sub = pd.read_csv("../input/sample_submission.csv")
print(sub.columns, sub.shape)
sub.head()


# In[29]:


sub['price'] = real_solution
sub.to_csv("mySolution.csv", index=False)  # 문제번호는 적지않고 id, price만으로 csv파일 만들기 


# In[30]:


model.intercept_


# In[35]:


model.coef_


# ### y = 319.840 x 'sqft_living' + -0.341 x 'sqft_lot' + -63708.721 x 'bedrooms' + 5357.139 x 'bathrooms'
# ### y = 319.840 x (x1) + -0.341 x (x2) + -63708.721 x (x3) + 5357.139 x (x4)

# In[36]:


## 소수점 표기를 소수점 세자리까지 보이게 하기
np.set_printoptions(formatter={'float_kind':lambda x:"{0:0.3f}".format(x)})

