import streamlit as st
import numpy as np
from gym_ui.model import DropoutModel
from gym_ui.deep_learning import deep_learning, machine_learning
from PIL import Image

image_path_1 = "image/노이탈(헬스)2.webp"
image_path_2 = "image/노이탈(헬스).webp"
image_path_3 = "image/이탈자(헬스).webp"
image_1 = Image.open(image_path_1)
image_2 = Image.open(image_path_2)
image_3 = Image.open(image_path_3)

st.title(':muscle: 헬스장 계약 종료 예측 :running:')
st.sidebar.header('입력 정보')
st.divider()


gender = st.sidebar.selectbox('성별', options=[0, 1])
near_Location = st.sidebar.selectbox('집에서 가까운지', options=[0, 1])
partner = st.sidebar.selectbox('회사에서 왔는지', options=[0, 1])
promo_friends = st.sidebar.selectbox('친구 소개로 왔는지 여부', options=[0, 1])
contract_period = st.sidebar.selectbox('계약기간', options=[1, 6, 12])
Group_visits = st.sidebar.selectbox('단체',  options=[0, 1])
age = st.sidebar.number_input('나이', min_value=18, max_value=120, value=30)
avg_additional_charges_total = st.sidebar.text_input('월 평균 추가 비용', value="0.0")
try:
    avg_additional_charges_total = float(avg_additional_charges_total)
except ValueError:
    st.sidebar.error("유효한 숫자를 입력하세요.")
# month_to_end_contract = st.sidebar.selectbox('계약 종료 월', options=range(1,13))
lifetime = st.sidebar.number_input('(월기준)다닌날', min_value=0, value=0)
avg_class_frequency_total = st.sidebar.text_input('월 평균 수업 참여 횟수', value="0.0")
try:
    avg_class_frequency_total = float(avg_class_frequency_total)
except ValueError:
    st.sidebar.error("유효한 숫자를 입력하세요.")
# avg_class_frequency_current_month = st.sidebar.text_input('이번달 수업 참가 횟수', value="0.0")
# try:
#     avg_class_frequency_current_month = float(avg_class_frequency_current_month)
# except ValueError:
#     st.sidebar.error("유효한 숫자를 입력하세요.")
# month_to_end_contract=contract_period
# avg_class_frequency_current_month=avg_class_frequency_total
input_data = np.array([[
    gender,
    near_Location,
    partner,
    promo_friends,
    contract_period,
    Group_visits,
    age,
    avg_additional_charges_total,
    # month_to_end_contract,
    lifetime,
    avg_class_frequency_total,
    # avg_class_frequency_current_month,
    ]])

deep_button = st.sidebar.button("딥러닝 모델로 예측")
machine_button = st.sidebar.button("머신러닝 모델로 예측")
if deep_button:
    result = deep_learning(input_data)  
    churn_probability = result  
    isTrue = churn_probability >= 0.5
    images = image_3 if isTrue else image_1
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header('예측 결과')
        st.subheader(f"예측결과: {'이탈' if isTrue else '노이탈'}")
        st.subheader(f'이탈 확률: {churn_probability * 100:.2f}%')
        st.subheader(f"{'이탈 가능성 높음' if isTrue else '이탈 가능성 낮음'}")
    with col2:
        st.image(images, width=200)
if machine_button:
    result = machine_learning(input_data) 
    churn_probability = result[1] * 100  
    isTrue = result[0] >= 0.5
    images = image_3 if isTrue else image_1
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header('예측 결과')
        st.subheader(f'예측결과: {"이탈" if isTrue else "노이탈"}')
        st.subheader(f'이탈 확률: {churn_probability[0]:.2f}%')
        st.subheader(f"{'이탈 가능성 높음' if isTrue else '이탈 가능성 낮음'}")
    with col2:
        st.image(images, width=200)