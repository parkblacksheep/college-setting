import streamlit as st

st.title("202302804 박종원 과제")

# 페이지 제목
st.header("페이지 제목")
# 페이지 소제목
st.subheader("페이지 소제목")
# 텍스트
st.text("텍스트")

# 이미지
from PIL import Image
image = Image.open('image.png')
st.image(image, caption='이미지 캡션')

# 데이터프레임
import pandas as pd
df = pd.read_csv('data.csv')
st.dataframe(df)

# 차트
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots()
sns.histplot(df['column'], ax=ax)
st.pyplot(fig)

# 입력
input_text = st.text_input("텍스트 입력", "기본값")

# 버튼
if st.button('버튼'):
# 버튼을 클릭하면 실행될 코드
# 라디오 버튼
    radio_button = st.radio("라디오 버튼", ('옵션 1', '옵션 2', '옵션 3'))
    
# 체크박스
if st.checkbox('체크박스'):
    # 체크박스를 클릭하면 실행될 코드
# 라디오 버튼
   radio_button = st.radio("라디오 버튼", ('옵션 1', '옵션 2', '옵션 3'))

# 셀렉트 박스
select_box = st.selectbox("셀렉트 박스", ['옵션 1', '옵션 2', '옵션 3'])

# 멀티 셀렉트 박스
multi_select_box = st.multiselect("멀티 셀렉트 박스", ['옵션 1', '옵션 2', '옵션 3'])

# 슬라이더
slider = st.slider("슬라이더", min_value=0, max_value=10, value=5, step=1)

# 날짜 선택
date = st.date_input("날짜 선택")

# 시간 선택
time = st.time_input("시간 선택")

#파일 업로드
uploaded_file = st.file_uploader("파일 업로드")

#페이지 간 구분
st.write("---")

#링크
st.markdown("[링크 텍스트](링크 URL)")

#블록
with st.beta_expander("블록 제목"):
# 블록 내용
  st.write("블록 내용")

#사이드바
st.sidebar.header("사이드바 제목")

#사이드바 입력
sidebar_input_text = st.sidebar.text_input("사이드바 입력", "기본값")

#사이드바 체크박스
if st.sidebar.checkbox('사이드바 체크박스'):
# 사이드바 체크박스를 클릭하면 실행될 코드
#사이드바 라디오 버튼
   sidebar_radio_button = st.sidebar.radio("사이드바 라디오 버튼", ('옵션 1', '옵션 2', '옵션 3'))

#사이드바 셀렉트 박스
sidebar_select_box = st.sidebar.selectbox("사이드바 셀렉트 박스", ['옵션 1', '옵션 2', '옵션 3'])

#사이드바 멀티 셀렉트 박스
sidebar_multi_select_box = st.sidebar.multiselect("사이드바 멀티 셀렉트 박스", ['옵션 1', '옵션 2', '옵션 3'])

#사이드바 슬라이더
sidebar_slider = st.sidebar.slider("사이드바 슬라이더", min_value=0, max_value=10, value=5, step=1)

#사이드바 날짜 선택
sidebar_date = st.sidebar.date_input("사이드바 날짜 선택")

#사이드바 시간 선택
sidebar_time = st.sidebar.time_input("사이드바 시간 선택")

#사이드바 파일 업로드
sidebar_uploaded_file = st.sidebar.file_uploader("사이드바 파일 업로드")
