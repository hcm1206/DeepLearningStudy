# 기계학습을 실제로 수행하려면 모든 데이터에 대한 손실함수 값을 구하여 정답과 비교해야 함
# 평균 손실 함수를 구할 때는 모든 데이터의 손실 함수 값을 구한 후 데이터의 수만큼 나눔

# 이 때 데이터의 수가 많을 때 데이터의 일부를 무작위로 뽑아 학습하는 것을 '미니배치(mini-batch) 학습'이라 함

import sys, os
# 파일 경로 오류로 인하여 절대경로 사용
sys.path.append("/Users/hyeon/Documents/Python/DeepLearningStudy")
import numpy as np
from dataset.mnist import load_mnist
# 훈련 데이터와 시험 데이터 불러오기(원-핫 인코딩)
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 훈련 데이터 이미지 행렬 형상 (60,000개 × 784(28×28)개 픽셀)
print(x_train.shape)
# 훈련 데이터 정답 레이블 행렬 형상 (60,000개 × 10(0~9)개의 정답 후보)
print(t_train.shape)

# 이 훈련데이터에서 무작위로 10장을 뽑기

# 훈련하는 전체 이미지 수(60,000)
train_size = x_train.shape[0]
# 무작위로 뽑을 (배치로 묶을) 이미지 수(10)
batch_size = 10
# 0 이상 train_size(60,000) 미만의 수 중에서 batch_size(10)개의 수를 무작위로 추출하여 batch_mask에 저장
batch_mask = np.random.choice(train_size, batch_size)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

# 0부터 60000 미만의 수 중에서 10개를 뽑아 넘파이 배열로 출력
print(np.random.choice(60000,10))


# 배치로 묶은 이미지도 입력받을 수 있는 교차 엔트로피 오차 함수 정의
def cross_entropy_error(y, t):
    # 입력받은 추론값의 차원이 1이라면(하나의 데이터(이미지)만 입력받았다면)
    if y.ndim == 1:
        # 정답값과 추론값 배열을 각각 1 × 10, 1 × 784 넘파이 배열로 변환
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # batch_size(배치 크기)를 추론값 배열의 0번째 차원(입력받은 데이터(이미지))의 수으로 지정 (일단 입력받은 모든 데이터(이미지)를 배치로 묶는 듯)
    batch_size = y.shape[0]
    # 데이터 하나의 교차 엔트로피 오차 계산을 배치크기만큼 반복하여 결과값을 모두 합한 후 배치크기만큼 나누면 평균 교차 엔트로피 오차 도출 가능
    return -np.sum(t*np.log(y+1e-7)) / batch_size



# 신경망의 추론값이 원-핫 인코딩이 아니고 정수로 출력될 경우 아래와 같이 교차 엔트로피 오차 함수 수정 가능
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t] + 1e-7))

    # 정답이 아닌 값은 모두 0으로 취급하므로 정답 레이블을 통해 정답인 인덱스만 계산하는 방식
    # np.arange(batch_size)는 0부터 batch_size 미만까지의 수를 1차원 배열로 저장 (ex 배치크기 5의 np.arange(batch_size) = [0,1,2,3,4])
    # t에는 정답인 값만 1차원 배열로 batch_size 크기만큼 저장되어있음 (ex. 배치크기 5의 t = [2,7,0,9,4])
    # 따라서 이 예시에서는 y[0,2], y[1,7], y[2,0], y[3,9], y[4,4]의 자연로그에 각각 delta(1e+7)을 더한 값들을 모두 합하여 음수를 붙인 값이 교차 엔트로피 오차 결과