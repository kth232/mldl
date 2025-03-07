
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

import numpy as np

# column_stack(): 전달받은 리스트를 일렬로 세운 후 차례대로 나란히 연결함
fish_data = np.column_stack((fish_length, fish_weight))

print(fish_data[:5])

# ones(): 원하는 개수만큼 1을 채운 배열 생성
# zeros(): 원하는 개수만큼 0을 채운 배열 생성
# concatenate(): 첫번째 차원을 따라 배열 연결
fish_target = np.concatenate((np.ones(35), np.zeros(14)))

print(fish_target)


# 사이킷런으로 훈련 세트와 테스트 세트 나누기
from sklearn.model_selection import train_test_split

# train_test_split(): 비율에 맞게 훈련 세트, 테스트 세트로 나눠주는 함수
# random_state 매개변수로 랜덤시드 지정
# stratify 매개변수에 타겟 데이터 전달하면 클래스 비율에 맞게 데이터 나눠줌
train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, stratify=fish_target, random_state=42)

# 넘파이 배열은 튜플로 표현
print(train_input.shape, test_input.shape)

print(train_target.shape, test_target.shape)

print(test_target)


from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)

print(kn.predict([[25, 150]]))

import matplotlib.pyplot as plt

# kneighbors(): 이웃까지의 거리와 이웃 샘플 인덱스를 반환
distances, indexes = kn.kneighbors([[25, 150]])

plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

print(train_input[indexes])

print(train_target[indexes])

print(distances)

plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlim((0, 1000)) # xlim(): x축의 범위를 지정하는 함수
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# axis=0으로 설정하여 행을 따라 각 열의 통계 값을 계산함
mean = np.mean(train_input, axis=0) # 평균 계산
std = np.std(train_input, axis=0) # 표준편차 계산

print(mean, std)

# 표준점수 = (원본 데이터 - 평균)/표준편차
train_scaled = (train_input - mean) / std

# 샘플도 평균과 표준편차를 이용해서 동일한 비율로 변환
new = ([25, 150] - mean) / std

plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 전처리한 데이터셋으로 훈련
kn.fit(train_scaled, train_target)

# 테스트 세트도 훈련 세트처럼 평균과 표준편차로 스케일 변환해야 함
test_scaled = (test_input - mean) / std

kn.score(test_scaled, test_target)

print(kn.predict([new]))

distances, indexes = kn.kneighbors([new])

plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexes,0], train_scaled[indexes,1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()