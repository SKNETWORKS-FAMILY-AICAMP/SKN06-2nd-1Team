import pandas as pd
import streamlit as st

def file_input(file_read, deep_learning, machine_learning_best_xgb):
    st.sidebar.header("파일 업로드")
    uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드하세요", type="csv")
    if uploaded_file:
        input_data = file_read(uploaded_file)
        deep_button = st.sidebar.button("딥러닝 모델로 예측")
        machine_button = st.sidebar.button("머신러닝 모델로 예측")
        method = ("deep_learning", "machine_learning")
        if deep_button:
            st.session_state["predictions"] = display(input_data, deep_learning, method[0])
        if machine_button:
            st.session_state["predictions"] = display(input_data, machine_learning_best_xgb,method[1])

    if "predictions" in st.session_state and st.session_state["predictions"] is not None:
        pageing(st.session_state["predictions"])
    else:
        st.info("예측 결과가 없습니다. CSV 파일을 업로드하고 예측 버튼을 눌러주세요.")

        
def display(input_data, predict_function, method):
    if method == "deep_learning":
        answer = []
        for i in range(input_data.shape[0]):
            dit = {}
            result = predict_function(input_data.iloc[[i]])  
            churn_probability = result  
            isTrue = churn_probability >= 0.5
            dit["예측결과"] = "이탈" if isTrue else "노이탈"
            dit["이탈확률"] = f"{churn_probability * 100:.2f}%"
            dit["이탈가능성"] = '이탈 가능성 높음🍕' if isTrue else '이탈 가능성 낮음💪'
            answer.append(dit)
    elif method == "machine_learning":
        answer = []
        for i in range(input_data.shape[0]):
            dit = {}
            xgb = predict_function(input_data.iloc[[i]])
            churn_probability = xgb[1] * 100  
            isTrue = xgb[0] >= 0.5
            dit["예측결과"] = "이탈" if isTrue else "노이탈"
            dit["이탈확률"] = f'{churn_probability[0]:.2f}%'
            dit["이탈가능성"] = '이탈 가능성 높음🍕' if isTrue else '이탈 가능성 낮음💪'
            answer.append(dit)

    return pd.DataFrame(answer)

def highlight_high_risk(row):
    return ['color: red' if "이탈 가능성 높음" == cell else '' for cell in row]
    
def pageing(df):
    
    if df.empty:
        st.warning("예측 결과 데이터가 없습니다.")
        return
        
    page_size = 10
    total_pages = (len(df) - 1) // page_size + 1
    page_number = st.slider(
        "페이지 번호를 선택하세요 :muscle:",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
        help=f"1에서 {total_pages} 사이의 페이지를 선택하세요."
    )
    start_idx = (page_number - 1) * page_size
    end_idx = min(start_idx + page_size, len(df))
    paginated_data = df.iloc[start_idx:end_idx]

    # 데이터 출력
    st.header("예측 결과")
    st.table(paginated_data)
    st.write(f"페이지 {page_number} / {total_pages}")