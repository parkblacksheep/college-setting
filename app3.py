import streamlit as st

st.title("202302804 박종원 과제")

# 페이지 제목
st.header("k-nn알고리즘을 적용한 iris분류, k-means clustering표시 ")
from scipy import signal
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import Input
from keras import Model
import matplotlib.pyplot as plt
import io
from contextlib import redirect_stdout

num_epochs = 100  # 초기 epoche 값 
data_np = np.loadtxt('./data.csv', dtype = float, delimiter = ',')    # loadtxt : 데이터를 불러오는 기능(txt, csv 등등등)
st.text('data_csv file -----------------')
st.write(data_np)
st.write(data_np.shape)
x = data_np[: , 0:1]  # X Value
y = data_np[: , 1:2]  # Y Value

# 두 개의 열을 생성합니다.
col1, col2 = st.columns(2)
# 첫 번째 열에 x 값을 표시합니다.
col1.header("X Values")
col1.write(x)
# 두 번째 열에 y 값을 표시합니다.
col2.header("Y Values")
col2.write(y)

fig = plt.figure(figsize = (5,2))
plt.plot(x,y)
st.pyplot(plt)

fig = plt.figure(figsize = (5,2))
plt.plot(x,y)
plt.scatter(x,y)
st.pyplot(plt)

st.write(x.shape)
st.write(x.shape[1:])

x1_input = Input(shape=(x.shape[1:]), name = 'x1_input')    # 입력층
x1_Dense_1 = layers.Dense(50, name = 'Dense_1')(x1_input)    # Dense : y = wx + b # 은닉층
x1_Dense_2 = layers.Dense(50, name = 'Dense_2')(x1_Dense_1)  # 은닉층
final = layers.Dense(1, name = 'final')(x1_Dense_2)          # 출력층
model = Model(inputs=x1_input, outputs = final)
model.compile(optimizer = 'adam', loss = 'mse')      # optimizer: 최적화 수식,  loss: 오차 수식
model.summary()

# 캡처할 출력을 위한 문자열 버퍼를 생성합니다.
buf = io.StringIO()        # 표준 출력을 버퍼로 재지향합니다.
with redirect_stdout(buf):
    model.summary()

st.text(buf.getvalue())   # 버퍼의 내용을 가져와 Streamlit에 표시합니다.
model_train = model.fit(x, y, epochs=st.sidebar.slider("epochs 횟수", min_value=0, max_value=200, value=100, step=5))

fig = plt.figure(figsize = (15,5))
plt.plot(model_train.history['loss'])
st.text("Model Train")
st.pyplot(plt)

prediction = model.predict([x])
fig = plt.figure(figsize = (15,5))
plt.plot(y, label='Actual')
plt.plot(prediction, label='Prediction')
# 범례를 추가합니다.
plt.legend()
st.text("Model Prediction")
st.pyplot(plt)

# Slide BAR 상기 model.fit 할때 eoches 값을 계산하여 그래프에 반영.
sidebar_date = st.sidebar.date_input("작성 날짜")
sidebar_time = st.sidebar.time_input("작성 시간")
fig.canvas.manager.full_screen_toggle()

#---------------------------------------------------------------------#

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

#데이터셋 로드
iris=load_iris()
X=iris.data
y=iris.target
#데이터셋 분리
X_train,X_test,Y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#KNN 알고리즘 적용
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)

#예측 결과
y_pred=knn.predict(X_test)
print("Predictions:",y_pred)

#정확도 계산
accuracy= np.mean(y_pred==y_test)
print("Accuracy:",accuracy)
#분류결과 시각화

fig = plt.figure(figsize = (15,5))
plt.scatter(X[:,0],X[:,1],c=y,cmap= 'viridis')
plt.xlabel('Sepal length')
plt.ylabel('sepal width')
st.text('K-NN 알고리즘을 적용한 Iris Classification분류 화면')

st.pyplot()

#----------------------------------------------------------------------------------------------------
#예시 데이터 생성
data=np.random.randn(100,2)

def initialize_centroids(data,k):
    centroids= data(np.random.choice(data.shape[0],k,replace=False))
    return centroids

