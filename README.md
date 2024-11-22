# SKN06-2nd-1Team
## 2차 단위 프로젝트 - 고객 이탈 예측 모델 / Customer Churn  </br>
조장 - :crown: 장예린 :crown: </br>
팀원 - 고성주, 박창규 </br>
</br>
## 1조 : 사다리
###
| 장예린 | 고성주 | 박창규 |
|:----------:|:----------:|:----------:|
| <img src="https://github.com/user-attachments/assets/1e75ad5d-9a1d-4314-b9b4-4ffc12e5e441" width="140" height="175" />  | <img src="https://github.com/user-attachments/assets/197ee150-7311-43e4-943e-6237a7151303" width="140" height="175" />  | <img src="https://github.com/user-attachments/assets/262cea2f-713e-4ac2-b07b-d97206846dee" width="140" height="175" /> |
| EDA, DL | ML | Streamlit |

</br></br></br>

##  헬스장 (GYM) 이탈 모델

### ✔️ 개발 기간
2024.11.13 ~ 2024.11.14 (총 2일)
</br>

### ✔️ 프로젝트 개요
</br>

>  현대인들은 건강과 체형 관리에 대한 관심이 증가하면서 헬스장의 이용이 늘어나고 있지만, 바쁜 일상과 다양한 운동 옵션으로 인해 헬스장에 대한 지속적인 이용이 어려워지고 있다.

>  특히, 헬스장 계약이 6개월 또는 1년 단위로 이루어지는 경우가 많고, 많은 고객들이 재계약을 하지 않거나 중도에 이탈하는 경향을 보인다. 이는 헬스장 운영에 큰 부담을 주며, 신규 고객 확보와 기존 고객 유지를 위한 전략적인 접근이 필요하하다.
</br></br>


### ✔️ 제공 데이터
gym_churn_us.csv : 캐글 자료 / 헬스장 (GYM) 이탈 데이터 </br>
14개 컬럼, 4000개 데이터 </br>
| 변수명                              | 설명                          | 값 또는 자료형         |
|-------------------------------------|-------------------------------|-----------------------|
| gender                              | 성별                          | 여자 0 / 남자 1        |
| Near_Location                       | 거리                          | 멀다 0 / 가깝다1      |
| Partner                             | 회사 할인 여부                | 개인 0 / 회사 할인 1    |
| Promo_friends                       | 지인소개 여부                | 없음 0 / 지인소개 1    |
| Phone                               | 연락처 제공 여부            | 제공X 0 / 제공O 1        |
| Contract_period                     | 계약 기간                     | int                   |
| Group_visits                        | 그룹세션 참여 여부          | No 0 / Yes 1          |
| Age                                 | 나이                          | int                   |
| Avg_additional_charges_total        | 추가사용료 평균              | float                 |
| Month_to_end_contract               | 남은 계약 기간               | float                   |
| Lifetime                            | 총 이용기간                  | int                   |
| Avg_class_frequency_total           | 평균 수업 참가               | float                 |
| Avg_class_frequency_current_month   | 이달 평균 수업 참가         | float     


</br>


### ✔️ Stacks
![Discord](https://img.shields.io/badge/discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=Git&logoColor=white)
![Github](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white)

![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white) 
![Jupyter](https://img.shields.io/badge/jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white) 

![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Pytorch](https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) 
![Scikit-Learn](https://img.shields.io/badge/scikitlearn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white) 
![Streamlit](https://img.shields.io/badge/streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) 



### ✔️ Requirements

jupyterlab==4.2.5
</br>
pandas==2.2.3
</br>
torch==2.5.1
</br>
scikit-learn==1.5.2
</br>
xgboost==2.1.2
</br>
matplotlib==3.9.2
</br>
seaborn==0.13.2
</br>
streamlit==1.39.0
</br>


### ✔️ 폴더트리
```
프로젝트
|-- Data 
|    |-- gym_churn_us.csv : 데이터 파일
| 
|-- Models
|    |-- best_gb.pkl : Machine Learning - Gradient Boosting 모델
|    |-- best_rf.pkl : Machine Learning - RandomForest 모델
|    |-- best_xgb.pkl : Machine Learning - XGBoosting 모델
|    |-- dout_model.pt : Deep Learning - Dropout 모델
|    |-- scaler.pkl : scaler 모델
|
|-- src
|    |-- image
|    |    |-- 노이탈(헬스).webp : 시각 자료
|    |    |-- 노이탈(헬스)2.webp : 시각 자료
|    |    |-- 이탈자(헬스).webp : 시각 자료
|    |-- models
|    |    |-- dropout_model.py : Dropout 모델 코드
|    |-- services
|    |    |-- deep_learning.py : Deep Learning 서비스 코드
|    |    |-- machine_learning.py : Machine Learning 서비스 코드
|    |-- ui
|    |    |-- direct_input : 직접입력 Streamlit
|    |    |-- file_input : 파일로 입력 Streamlit 
|    |---utils
|    |    |-- file_read : csv파일 처리
|    |    |-- gym_img : 이미지 처리
|
|-- Deep_Learning.ipynb : 딥러닝 진행 파일
|-- EDA.ipynb : EDA 진행 파일
|-- Machine_Learning.ipynb : 머신러닝 진행 파일
|-- app.py : Streamlit 실행 파일
|
|-- readme.md : readme 파일
|-- requirements.txt : 설치 모듈 리스트
```
</br></br>

### ✔️ 데이터 전처리 
❗️ EDA ❗️</br>
</br>

> - 결측치 확인 :  결측치 없음 (14개 컬럼 / 0 Null)</br>
> ![스크린샷 2024-11-14 150900](https://github.com/user-attachments/assets/390899e5-f850-4f24-a522-a999207884f3)</br>
>
> - 전체 데이터 중 이탈값의 분포</br>
> ![image](https://github.com/user-attachments/assets/9d30fe05-9959-4c33-9088-2ad49246b116)</br>
>
>  - 이탈값에 따른 범주형 변수 그래프</br>
> ![image](https://github.com/user-attachments/assets/aaee7505-cd6f-4823-a096-cc4ef2b1dd3b)</br>
>
> - 이탈값에 따른 숫자형 변수 그래프</br>
> ![image](https://github.com/user-attachments/assets/e4867480-1e5c-4d97-85a5-574d45f9f2de)</br>
>
> - 이상치 확인: 이상치 없음</br>
>
> 
> - 모든 변수들의 상관관계</br>
> ![image](https://github.com/user-attachments/assets/d8c2914d-ea58-4a19-b164-7fa775dfa032)</br>
>
> - 상관관계가 높은 변수는 서로 중복된 정보를 제공한다.
> - 그에 따라 Avg_class_frequency_total, Avg_class_frequency_current_month 중 Avg_class_frequency_total  </br>
>   Contract_period, Month_to_end_contract 중 Contract_period 만 사용하는걸로 결정
> 
> - 이탈값과 다른 변수들의 상관관계</br>
> ![image](https://github.com/user-attachments/assets/8f6c5af3-b994-46e0-9a4a-6044f3969363)</br>
> - 상관관계가 거의 작고, 등록 시 필수 입력값이 아닌 Phone (연락처 보유 여부)는 제외하고 학습 진행
> - gender (성별)도 상관관계는 작으나, 실제로는 유의미한 데이터를 제공할 확률이 크기 때문에 포함하여 진행
> - 총 13개의 변수 (churn 제외) 중 10개의 변수를 사용하여 학습 및 예측을 진행
</br></br>
 
### ✔️ 모델학습
❗️ Machine Learning❗️ </br>
   Gradient Boosting / Random Forset / KNN / XGBoosting 으로 베이스라인 모델 학습</br>
   ![image](https://github.com/user-attachments/assets/36c5893e-c517-48d2-94d9-9fcc5edc6a74)
</br>
   </br>
   가장 낮은 KNN 제외 3개 모델로 Grid 또는 Randomized Search 진행</br>
   - RandomForest : 42개 조합으로 Grid Search 진행</br>
![image](https://github.com/user-attachments/assets/297cfd28-d67a-4b43-ae4f-71cad080898f)
</br>
   </br>
   
   - Gradient Boosting : 600개 조합으로 Randomized Search 진행</br>
![image](https://github.com/user-attachments/assets/a36771aa-01a0-48f7-ac0e-ed9abe8576a5)
</br>
   </br>
   
   - XG Boosting : 600개 조합으로 Randomized Search 진행</br>
![image](https://github.com/user-attachments/assets/68d57dba-b358-4749-9207-04af57e6afef)
</br>
   </br>

   - Lifetime, Contract Period 순으로 이탈 확률에 많은 영향을 끼치고 있음</br>
![image](https://github.com/user-attachments/assets/481e77ee-096a-4395-9073-bcd2f2509bd7)
</br>
   </br></br>

❗️ Deep Learning❗️ </br>

> - 최적 파라미터 사용을 위한 dropout 및 ealry stopping 사용 </br>
>
>   ![image](https://github.com/user-attachments/assets/4d64e529-36d3-43e8-8560-a0a94875ccd3) </br>
>
>
>   ![image](https://github.com/user-attachments/assets/25533775-0d44-4299-bb0a-e6b1c8ba010f) </br>

> - 모델 학습 결과</br>
> ![image](https://github.com/user-attachments/assets/60ec35a2-63fa-47b7-8d04-2f84da67377f)
> ![image](https://github.com/user-attachments/assets/ac85d6e3-3d9a-4a99-9fa7-10a48d06a228)
> ![image](https://github.com/user-attachments/assets/7c15458c-6096-4dad-80f7-38a3b0b1958a)


> </br>




   > --
   > 
</br></br>

### ✔️ Streamlit
> -- 직접 입력
> 
> <img src="https://github.com/user-attachments/assets/fe2815f4-0d47-4a84-89ef-427dcacc51e4" alt="Main" width="800">
> <img src="https://github.com/user-attachments/assets/ff89b770-13d6-45c5-8f7b-575801157313" alt="Main2" width="800">

> -- 파일 입력
> 
> <img src="https://github.com/user-attachments/assets/e999c55b-31cf-48ec-a1b5-a1f29ebd7489" alt="Main_file" width="800">
> <img src="https://github.com/user-attachments/assets/03fe21c9-b20c-4c9d-a765-5d0852dbc223" alt="Main_file2" width="800">

</br></br>


### ✔️ 팀원 회고

:crown: 장예린 :crown:
> 딥러닝을 진행할때 고려해야할 것들이 많았는데, 다양한 도전을 해보지 못 해 아쉬움이 남는다.  </br>
> 다음에는 마음의 여유를 가지고 다양한 시도를 해보고 싶다. 
> 
고성주
> 제공된 데이터에 결측치나 이상치가 없어서인지, 베이스모델만 돌렸을때도 결과가 너무 좋게 나왔다.</br>
> 그래서인지, 모델 개선을 위해 많은 고민을 하지 않은듯하여 아쉬움이 남는다. 
> 
박창규
> 지식의 부족해 다양한 방법을 시도하지 못한 것이 아쉬웠다.
> 
>
