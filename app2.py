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
from contextlib import redirect_stdout
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import random
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


#---------------------------------------------------------------------#

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

st.pyplot(fig)

#----------------------------------------------------------------------------------------------------
#예시 데이터 생성

# 예제 데이터 생성
np.random.seed(100)
num_data=50 

x11=np.linspace(0.3,0.7,20)
x12=np.linspace(1.3,1.8,15)
x13=np.linspace(2.4,3,15)
x1=np.concatenate((x11,x12,x13),axis=None)
error=np.random.normal(1,0.5,num_data)
x2=1.5*x1+2+error
fig=plt.figure(figsize=(7,7))
fig.set_facecolor('white')
plt.scatter(x1,x2,color='k')

def kmeans_clustering(X, n_clusters, init_center=None, max_iter=10, epsilon=1e-4, random_state=100): 
    # inititalize centeroids
    if init_center is None:
        random.seed(random_state)
        idx = random.sample(range(X.shape[0]), n_clusters)
        center = X[idx,:]
    else:
        center = init_center
    iteration = 1
    
    labels_history = [] # label history 
    center_history = [] # centeroid history
    while(iteration<=max_iter):
        ## assign label
        labels = []
        for i in range(X.shape[0]):
            data = X[i, :]
            labels.append(np.argmin([np.linalg.norm(data-x) for x in center]))
        
        labels = np.array(labels)
        ## update centeroids
        next_center = []
        for i in range(n_clusters):
            target_idx = np.where(labels==i)[0]
            center_val = np.mean(X[target_idx,:], axis=0)
            next_center.append(center_val)
 
        next_center = np.array(next_center)
        if epsilon:
            if np.linalg.norm(next_center-center) <= epsilon:
                break
        center = next_center
        labels_history.append(labels)
        center_history.append(center)
        iteration += 1
    return (labels, iteration, labels_history, center_history)  

X = np.stack([x1, x2], axis=1)
 
init_center= np.array([[2,4],[1,5],[2.5,6]])
max_iter=50
epsilon=1e-10
random_state=101
n_clusters=3
results = kmeans_clustering(X, n_clusters, init_center, max_iter, epsilon=1e-4, 
                           random_state=100)
labels = results[0]

fig = plt.figure(figsize=(7,7))
fig.set_facecolor('white')
for i, label in enumerate(labels):
    if label == 0:
        color = 'blue'
    elif label ==1:
        color = 'red'
    else:
        color = 'green'
    plt.scatter(X[i,0],X[i,1], color=color)
    
plt.xlabel('x1')
plt.ylabel('x2')
st.text('K-means clusering을 표시')
st.pyplot(fig)

 
