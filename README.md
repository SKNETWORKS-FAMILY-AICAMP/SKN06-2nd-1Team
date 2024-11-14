# SKN06-2nd-1Team
## 2차 단위 프로젝트 - 고객 이탈 예측 모델 / Customer Churn  </br>
조장 - :crown: 장예린 :crown: </br>
팀원 - 고성주, 박창규 </br>

## 1조
###
| 장예린 | 고성주 | 박창규 |
| -- | -- | -- |
| -- | -- | -- |
| -- | -- | -- |

</br></br></br>

##  헬스장 (GYM) 이탈 모델

### ✔️ 개발 기간
2024.11.13 ~ 2024.11.14 (총 2일)
</br>

### ✔️ 프로젝트 개요
</br>

>  현대인들은 건강과 체형 관리에 대한 관심이 증가하면서 헬스장의 이용이 늘어나고 있지만, 바쁜 일상과 다양한 운동 옵션으로 인해 헬스장에 대한 지속적인 이용이 어려워지고 있습니다.

>  특히, 헬스장 계약이 6개월 또는 1년 단위로 이루어지는 경우가 많고, 많은 고객들이 재계약을 하지 않거나 중도에 이탈하는 경향을 보입니다. 이는 헬스장 운영에 큰 부담을 주며, 신규 고객 확보와 기존 고객 유지를 위한 전략적인 접근이 필요합니다.
</br>


### ✔️ 제공 데이터
gym_churn_us.csv : 캐글 자료 / 헬스장 (GYM) 이탈 데이터 </br>
> gender : 성별 (여자 0 / 남자 1) </br>
> Near_Location : 거리 (가깝다 0 / 멀다 1) </br>
> Partner : 회사지원 여부 (개인 0 / 회사지원 1) </br>
> Promo_friends : 지인소개 여부 (없음 0 / 지인소개 1) </br>
> Phone : 연락처 보유 (없음 0 / 보유 1) </br>
> Contract_period : 계약 기간 (int) </br>
> Group_visits : 그룹세션 (No 0 / Yes 1) </br>
> Age : 나이 (int) </br>
> Avg_additional_charges_total : 추가사용료 평균 (float) </br>
> Month_to_end_contract : 남은 계약 기간 (int) </br>
> Lifetime : 총 이용기간 (int) </br>
> Avg_class_frequency_total : 평균 수업 참가 (float) </br>
> Avg_class_frequency_current_month : 이달 평균 수업 참가 (float) </br>


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
| |-- Deep_Learning.ipynb : 딥러닝 진행 파일
|-- EDA.ipynb : EDA 진행 파일
|-- Machine_Learning.ipynb : 머신러닝 진행 파일
|
|-- readme.md : readme 파일
|-- requirements.txt : 설치 모듈 리스트
```
### ✔️ 데이터 전처리 
❗️ EDA ❗️</br>
</br>

> - 결측치 확인 :  결측치 없음 
> 
> ![스크린샷 2024-11-14 150900](https://github.com/user-attachments/assets/390899e5-f850-4f24-a522-a999207884f3)



> - 전체 데이터 중 이탈값의 분포
>
> ![image](https://github.com/user-attachments/assets/9d30fe05-9959-4c33-9088-2ad49246b116)
>
>  - 이탈값에 따른 범주형 변수 그래프
>    
> ![image](https://github.com/user-attachments/assets/aaee7505-cd6f-4823-a096-cc4ef2b1dd3b)
>
> - 이탈값에 따른 숫자형 변수 그래프
>   
> ![image](https://github.com/user-attachments/assets/1ffb4f9a-4908-4bc8-ba6a-e7fa52f6039d)
>
> 
> - 이상치 확인: 이상치 않음


> -  모든 변수들의 상관관계
>   
> ![image](https://github.com/user-attachments/assets/d8c2914d-ea58-4a19-b164-7fa775dfa032)
>
> - 이탈값과 다른 변수들의 상관관계
>   
> ![image](https://github.com/user-attachments/assets/8f6c5af3-b994-46e0-9a4a-6044f3969363)
>
> - 상관관계가 거의 작고, 등록 시 필수 입력값이 아닌 Phone (연락처 보유 여부)는 제외하고 학습 진행
> - gender (성별)도 상관관계는 작으나, 실제로는 유의미한 데이터를 제공할 확률이 크기 때문에 포함하여 진행






</br>
 
### ✔️ 모델학습
❗️ Machine Learning❗️ </br>
   Gradient Boosting / Random Forset / KNN / XGBoosting 으로 베이스라인 모델 학습
   > ![image](https://github.com/user-attachments/assets/7a35034b-ba86-493c-a8ec-f6a2a68f1b50)
   -> 가장 낮은 KNN 제외 3개 모델로 Grid 또는 Randomized Search 진행
   >![image](https://github.com/user-attachments/assets/a2ae3675-4336-4fab-8bdb-6265a3f7f5ab)
   > ![image](https://github.com/user-attachments/assets/500b1ecc-4f4a-4ce8-a8ab-91df350e56fd)
   >![image](https://github.com/user-attachments/assets/6aef44de-7cc7-43d1-9c77-10ac884c44d6)
> 
   Lifetime, Contract Period 순으로 이탈 확률에 많은 영향을 끼치고 있음
> 
   > ![image](https://github.com/user-attachments/assets/6a2fa6ba-4732-4c8d-9d13-b73b21fa1695)
</br>
❗️ Deep Learning❗️ </br>
> - 최적 파라미터 사용을 위한 dropout 및 ealry stopping 사용
>   
> 

> - 모델 학습 결과

> ![스크린샷 2024-11-14 123131](https://github.com/user-attachments/assets/5ec3e97b-c42c-4f3c-bac1-ac5358456b84)




   > --
   > 
</br>

### ✔️ Streamlit
   > --
   > --
   > 
</br>


### ✔️ 팀원 회고

:crown: 장예린 :crown:
> --
> --
> 
고성주
> --
> --
> 
박창규
> --
> --
>
