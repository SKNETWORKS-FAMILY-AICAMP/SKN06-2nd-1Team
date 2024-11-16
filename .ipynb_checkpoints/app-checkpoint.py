import streamlit as st
import numpy as np
import pandas as pd
from src.models.dropout_model import DropoutModel # 가중치를 따로 저장해주지 않아서
from src.services.deep_learning import deep_learning
from src.services.machine_learning import machine_learning_best_xgb
from src.utils.file_read import file_read
from src.utils.gym_img import image
from src.ui.direct_input import direct_input
from src.ui.file_input import file_input
image = image()

st.title(':muscle: 헬스장 계약 종료 예측 :running:')
st.sidebar.header('입력 정보')
input_method = st.sidebar.selectbox("데이터 입력 방식 선택", options=["직접 입력", "파일로 입력"])
st.divider()

############################## 값 입력 처리 ##############################
if input_method == "직접 입력":
    direct_input(deep_learning, machine_learning_best_xgb, image)
############################## ############################## ##############################
elif input_method == "파일로 입력":
    file_input(file_read,deep_learning, machine_learning_best_xgb)
