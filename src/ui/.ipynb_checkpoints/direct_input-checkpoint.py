import numpy as np
import streamlit as st

def direct_input(deep_learning, machine_learning_best_xgb, image):
    gender = st.sidebar.selectbox('성별', options=["남자", "여자"])
    near_Location = st.sidebar.selectbox('집 근처 여부', options=["O", "X"])
    partner = st.sidebar.selectbox('회사 소개 여부', options=["O", "X"])
    promo_friends = st.sidebar.selectbox('친구 소개 여부', options=["O", "X"])
    contract_period = st.sidebar.selectbox('계약기간', options=[1, 6, 12])
    Group_visits = st.sidebar.selectbox('단체 여부', options=["O", "X"])
    age = st.sidebar.number_input('나이', min_value=18, max_value=120, value=30)
    avg_additional_charges_total = st.sidebar.slider('월 평균 추가 비용$', 0.001, 800.00)
    lifetime = st.sidebar.number_input('(월기준) 다닌날', min_value=0, value=0)
    avg_class_frequency_total = st.sidebar.slider('월 평균 수업 참여 횟수', 0.001, 7.00)
    
    gender = 0 if gender == "남자" else 1
    near_Location = 0 if near_Location == "X" else 1
    partner = 0 if partner == "X" else 1
    promo_friends = 0 if promo_friends == "X" else 1
    Group_visits = 0 if Group_visits == "X" else 1

    input_data = np.array([[
        gender,
        near_Location,
        partner,
        promo_friends,
        contract_period,
        Group_visits,
        age,
        avg_additional_charges_total,
        lifetime,
        avg_class_frequency_total,
    ]])
    # Buttons
    deep_button = st.sidebar.button("딥러닝 모델로 예측")
    machine_button = st.sidebar.button("머신러닝 모델로 예측")

    method = ("deep_learning", "machine_learning")
    if deep_button:
        result = deep_learning(input_data)
        render_result(result, image, method[0])
    if machine_button:
        xgb = machine_learning_best_xgb(input_data)
        render_result(xgb, image, method[1])


def render_result(churn_probability, image, method):
    if method == "deep_learning":
        isTrue = churn_probability >= 0.5
        images = image[1] if isTrue else image[0]
        result = churn_probability*100
    elif method == "machine_learning":
        isTrue = churn_probability[0] >= 0.5
        images = image[1] if isTrue else image[0]
        result = float(churn_probability[1] * 100)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header('예측 결과')
        st.subheader(f"예측결과: {'이탈' if isTrue else '노이탈'}")
        st.subheader(f'이탈 확률: {result:.2f}%')
        st.subheader(f"{'이탈 가능성 높음' if isTrue else '이탈 가능성 낮음'}")
    with col2:
        st.image(images, width=200)
