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

</br>


### ✔️ 제공 자료
1. --
> --

2. --
> --

3. --
> --

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
--
>![image](https://github.com/user-attachments/assets/9d30fe05-9959-4c33-9088-2ad49246b116)
 전체 데이터 중 이탈값의 분포
> ![image](https://github.com/user-attachments/assets/aaee7505-cd6f-4823-a096-cc4ef2b1dd3b)
 이탈값에 따른 범주형 변수 그래프
> ![image](https://github.com/user-attachments/assets/1ffb4f9a-4908-4bc8-ba6a-e7fa52f6039d)
 이탈값에 따른 숫자형 변수 그래프
> ![image](https://github.com/user-attachments/assets/d8c2914d-ea58-4a19-b164-7fa775dfa032)
 모든 변수들의 상관관계
> ![image](https://github.com/user-attachments/assets/8f6c5af3-b994-46e0-9a4a-6044f3969363)

 이탈값과 다른 변수들의 상관관계
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
    최적 파라미터 사용을 위한 dropout 및 ealry stopping 사용한 모델 학습 결과
> ![스크린샷 2024-11-14 124753](https://github.com/user-attachments/assets/90552a15-db4a-4fa7-8c93-df7ad7a968a1)


|-- Deep_Learning.i과
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
