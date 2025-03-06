# 생선 분류

# 그래프 그리는 패키지 amtplotlib import
#plt: pyplot 함수의 관용적 줄임말
import matplotlib.pyplot as plt

# 도미 데이터
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 방어 데이터
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 산점도 그리는 scattoer 함수(x축, y축)
# plt.scatter(bream_length, bream_weight)
# plt.xlabel('length') # x축=길이
# plt.ylabel('weight') # y축=무게
# plt.show()

# 도미와 빙어 2개의 산점도를 한 그래프로 그리기
# plt.scatter(bream_length, bream_weight)
# plt.scatter(smelt_length, smelt_weight)
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

# 두 리스트를 더해서 하나의 리스트로 생성
length = bream_length+smelt_length
weight = bream_weight+smelt_weight

# zip함수를 사용해서 2차원 리스트 생성
# zip(): 나열된 리스트에서 하나씩 원소를 꺼내 반환환
fish_data = [[l, w] for l, w in zip(length, weight)]

# print(fish_data)

# 정답 데이터 생성(1=도미, 0=빙어)
fish_target = [1]*35 + [0]*14
# print(fish_target)

# 사이킷런 패키지에서 k최근접이웃 알고리즘을 구현한 클래스만 import
from sklearn.neighbors import KNeighborsClassifier

# 클래스 객체 생성
kn = KNeighborsClassifier()

# fit(): 주어진 데이터로 알고리즘 훈련
kn.fit(fish_data, fish_target)

# score(): 모델 평가, 1에 가까울수록 정확도 높음
kn.score(fish_data, fish_target)


# plt.scatter(bream_length, bream_weight)
# plt.scatter(smelt_length, smelt_weight)
# plt.scatter(30, 600, marker='^')
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

# predict(): 새로운 데이터의 정답을 예측하는 메서드, 2차원 리스트 필요
kn.predict([[30, 600]])

# x, y 속성에 생선 데이터, 정답 데이터를 가지고 있음
# print(kn._fit_X)
# print(kn._y)


# 기본값을 49개로 설정 = 참고데이터를 49개로 함
# kn49 = KNeighborsClassifier(n_neighbors=49)

# kn49.fit(fish_data, fish_target)
# kn49.score(fish_data, fish_target)
# print(35/49)

kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)


for n in range(5, 50):
    # 최근접 이웃 개수 설정
    kn.n_neighbors = n
    # 점수 계산
    score = kn.score(fish_data, fish_target)
    # 100% 정확도에 미치지 못하는 이웃 개수 출력
    if score < 1:
        print(n, score)
        break