# 판다스: 데이터 분석 라이브러리,pd 별칭 관례적 사용
import pandas as pd
import numpy as np

# 판다스로 csv 파일 읽기
df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy() # 넘파이 배열로 변환
print(perch_full)

perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
     1000.0, 1000.0]
     )


from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)

# 사이킷런 변환기
from sklearn.preprocessing import PolynomialFeatures

# 훈련 후 변환해야 함
# poly = PolynomialFeatures()
poly = PolynomialFeatures(include_bias=False) # 절편을 위한 항 제거
poly.fit([[2, 3]])
print(poly.transform([[2, 3]]))

poly = PolynomialFeatures(include_bias=False)

poly.fit(train_input)
train_poly = poly.transform(train_input)
print(train_poly.shape)

# get_feature_names_out(): 특성이 각각 어떤 조합으로 만들어졌는지 알려주는 메서드
poly.get_feature_names_out()

# 테스트 세트 변환
test_poly = poly.transform(test_input)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))

# degree 매개변수로 5제곱까지 특성 추가
poly = PolynomialFeatures(degree=5, include_bias=False)

poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

print(train_poly.shape)

lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))

# 규제 전 표준점수로 변환
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_poly)

# 훈련 세트로 학습한 변환기로 테스트 세트까지 변환해야 함
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# 릿지
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

import matplotlib.pyplot as plt

# 알파값이 바뀔 때마다 score 결과를 저장할 리스트
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # 릿지 모델을 만든다
    ridge = Ridge(alpha=alpha)
    # 릿지 모델을 훈련한다
    ridge.fit(train_scaled, train_target)
    # 훈련 점수와 테스트 점수를 저장한다
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))

# alpha값을 동일한 간격으로 표현하기 위해 로그함수로 바꾸어 지수로 표현
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

# 최적의 alpha값으로 훈련한 경우
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)

print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

# 라쏘
from sklearn.linear_model import Lasso

lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))

train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # 라쏘 모델을 만든다, 반복 횟수를 늘리기 위해 max_iter 매개변수 값 지정함
    lasso = Lasso(alpha=alpha, max_iter=10000)
    # 라쏘 모델을 훈련한다
    lasso.fit(train_scaled, train_target)
    # 훈련 점수와 테스트 점수를 저장한다
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))

# alpha값을 동일한 간격으로 표현하기 위해 로그함수로 바꾸어 지수로 표현
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

# 최적의 alpha값으로 훈련한 경우
lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)

print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))

# coef_ 속성에 계수 값 저장됨
print(np.sum(lasso.coef_ == 0))