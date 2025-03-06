fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]


fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
fish_target = [1]*35 + [0]*14


from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()

# 훈련은 0~34번째 데이터만 슬라이싱
train_input = fish_data[:35]
train_target = fish_target[:35]

# 테스트는 35~49번째 데이터만 슬라이싱
test_input = fish_data[35:]
test_target = fish_target[35:]

# 샘플링 편향
kn.fit(train_input, train_target)
kn.score(test_input, test_target)


# 넘파이 라이브러리 import
import numpy as np

# 리스트를 넘파이 배열로 변경 = 넘파이 array()에 리스트 전달
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

# 넘파이는 배열의 차원 구분이 쉽도록 행과 열을 가지런히 출력해줌
print(input_arr)

# 배열 크기 알려주는 shape 속성
print(input_arr.shape)

# 시드값이 같으면 난수 생성해도 같은 난수 값 추출됨
np.random.seed(42)
# 0~48까지 1씩 증가하는 배열을 만듦
index = np.arange(49)
#주어진 배열을 무작위로 섞음
np.random.shuffle(index)

# print(index)

# 0~34번째 데이터는 훈련 세트
train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

print(input_arr[13], train_input[0])

# 35~48번째 데이터는 테스트 세트
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

import matplotlib.pyplot as plt

# plt.scatter(train_input[:, 0], train_input[:, 1])
# plt.scatter(test_input[:, 0], test_input[:, 1], c='red')
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

# k-최근접 이웃 모델 훈련
kn.fit(train_input, train_target)

kn.score(test_input, test_target)

kn.predict(test_input)

test_target