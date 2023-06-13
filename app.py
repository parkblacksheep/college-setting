import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def iris_classification():
    # Iris 데이터셋 로드
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # k-NN 분류기 초기화 및 학습
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = knn.predict(X_test)

    # 정확도 평가
    accuracy = accuracy_score(y_test, y_pred)

    # 결과 출력
    st.subheader("Iris Classification")
    st.write("Accuracy:", accuracy)

def initialize_centroids(data, k):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    return centroids

def kmeans_clustering():
    # 예시 데이터 생성
    data = np.random.randn(100, 2)

    # 클러스터 개수 입력
    k = st.slider("Select the number of clusters", min_value=1, max_value=10)

    # k-means 클러스터링 수행
    centroids = initialize_centroids(data, k)
    kmeans = KMeans(n_clusters=k, init=centroids, random_state=42)
    kmeans.fit(data)
    labels = kmeans.predict(data)

    # 클러스터 시각화
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100)
    plt.title("k-means Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    st.pyplot()

    # 클러스터 중심 및 할당된 레이블 출력
    st.subheader("Cluster Centers:")
    st.write(centroids)

    st.subheader("Assigned Labels:")
    st.write(labels)

# Streamlit 애플리케이션 설정
def main():
    st.title("Machine Learning Examples")

    # Iris Classification 예제 실행
    iris_classification()

    # K-means Clustering 예제 실행
    kmeans_clustering()

if __name__ == '__main__':
    main()
