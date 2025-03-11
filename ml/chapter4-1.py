import pandas as pd

fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head() # 처음 5개 행만 출력

# species 열에서 고유한 값 추출
print(pd.unique(fish['Species']))

# 넘파이 배열로 변환
fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()

print(fish_input[:5])

fish_target = fish['Species'].to_numpy()

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42)

# 표준화 전처리
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# k-최근접 이웃 분류기로 확률 예측
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(n_neighbors=3) # 이웃 개수는 3으로 지정
kn.fit(train_scaled, train_target)

print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

# 정렬된 타겟값은 classes_ 속성에 저장됨
print(kn.classes_)

# 예측값을 타겟값으로 출력
print(kn.predict(test_scaled[:5]))


import numpy as np

# predict_proba(): 클래스별 확률값 반환
proba = kn.predict_proba(test_scaled[:5])
# round()는 기본적으로 소수점 첫번째 자리에서 반올림함, decimals 매개변수로 유지할 자릿수를 지정
print(np.round(proba, decimals=4))

# 슬라이싱 연산자로 2차원 배열 입력
distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])


import matplotlib.pyplot as plt

# 시그모이드 함수 계산
z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))

plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()

# 이진 분류를 위해 bream과 smelt만 골라냄
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

# 로지스틱 회귀-이진 분류
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

print(lr.predict(train_bream_smelt[:5]))
print(lr.predict_proba(train_bream_smelt[:5]))

print(lr.classes_)

# 로지스틱 회귀가 학습한 계수
print(lr.coef_, lr.intercept_)

# z값 출력
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)

# 시그모이드 함수
from scipy.special import expit

print(expit(decisions))

# 로지스틱 회귀-다중 분류
# 충분한 반복을 위해 횟수를 1000번으로 지정
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

print(lr.predict(test_scaled[:5]))

proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))

print(lr.classes_)

print(lr.coef_.shape, lr.intercept_.shape)

# 각 샘플의 클래스별 z값 계산
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))

# 소프트맥스 함수
from scipy.special import softmax

# axis=1로 지정해서 각 행에 대해 소프트맥스를 계산함
proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))
