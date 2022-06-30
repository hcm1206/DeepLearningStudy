# MNIST 데이터셋을 이용하여 데이터를 추론하는 신경망 구현
# 입력층 뉴런 784개(이미지 픽셀 수), 출력층 뉴런 10개(0~9 사이의 숫자 판정)
# 은닉층 2개 (1번째 은닉층에 임의로 50개 뉴런, 2번째 은닉층에 임의로 100개 뉴런 배치)

import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from mnist import load_mnist

# 활성화 함수로 시그모이드 함수 사용
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# MNIST 데이터셋에서 시험용 이미지 객체, 시험용 이미지 레이블 불러오기
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

# sample_weight 피클 파일에서 미리 설정된 가중치 매개변수를 딕셔너리로 불러오기
def init_network():
    with open("Chap3. 신경망\sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

# 신경망 구현
def predict(network, x):
    # 가중치를 불러와 W1, W2, W3에 저장
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    # 편향을 불러와 b1, b2, b3에 저장
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 입력층에서 첫번째 은닉층으로의 계산(입력층과 가중치(1)를 행렬곱하고 편향을 더함)값을 a1에 저장
    a1 = np.dot(x, W1) + b1
    # 첫번째 계산 결과(a1)을 활성화 함수에 입력하여 출력값을 첫번째 은닉층(z1)에 저장
    z1 = sigmoid(a1)
    # 첫번째 은닉층에서 두번째 은닉층으로의 계산(첫번째 은닉층과 가중치(2)를 행렬곱하고 편향을 더함)값을 a2에 저장
    a2 = np.dot(z1, W2) + b2
    # 두번쨰 계산 결과(a2)를 활성화 함수에 입력하여 출력값을 두번째 은닉층(z2)에 저장
    z2 = sigmoid(a2)
    # 두번째 은닉층에서 출력층으로의 계산(두번째 은닉층과 가중치(3)를 행렬곱하고 편향을 더함)값을 a3에 저장
    a3 = np.dot(z2, W3) + b3
    # 세번째 계산 결과(a3)를 활성화 함수에 입력하여 출력값을 출력층(y)에 저장
    y = sigmoid(a3)

    return y

# 시험용 이미지 객체, 시험용 이미지 레이블을 각각 x, t에 저장
x, t = get_data()
# network에 가중치 매개변수가 저장된 딕셔너리 저장
network = init_network()

# 정확도를 계산할 변수 선언 (정답 횟수)
accuracy_cnt = 0
# 시험용 이미지 수만큼 반복(10,000번)
for i in range(len(x)):
    # [i]번째 시험용 이미지 객체(행렬)을 신경망에 입력하여 출력값을 y에 저장
    y = predict(network, x[i])
    # y 출력값(배열)에서 각 원소 중 가장 큰(확률이 높은) 원소의 인덱스 번호를 p에 저장
    p = np.argmax(y)
    # p가 [i]번째 인덱스 시험용 이미지 레이블(정답인 숫자)와 같다면 정확도 계산 변수를 1 올림
    if p == t[i]:
        accuracy_cnt += 1

# 정확도 변수를 이미지 전체 수(10,000)으로 나누어 정확도를 확률로 계산
print("Accuracy:" + str(float(accuracy_cnt)/len(x)))


# load_mnist 함수의 인수인 normalize를 True로 설정하여 정규화 진행
# 데이터를 원하는 범위의 값으로 변환하는 과정을 '정규화'라 하고 신경망 입력 데이터에 특정 변환을 가하는 것을 '전처리'라 함

print()

# 이미지 하나를 처리할 때 거쳐가는 신경망에서의 행렬 곱 확인

# 정보 불러오기
x, _ = get_data()
network = init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']

# 시험용 이미지 객체의 행렬 형상 출력 (10000개, 23×23 = 784의 픽셀 수(이미지 크기))
print(x.shape)
# 0번째 시험용 이미지 객체 1개의 행렬 형상 출력 (23×23 = 784개의 픽셀 수)
print(x[0].shape)
# 1번째 가중치 행렬 형상 출력 (784×50)
print(W1.shape)
# 2번째 가중치 행렬 형상 출력 (50×100)
print(W2.shape)
# 3번째 가중치 행렬 형상 출력 (100×10)
print(W3.shape)
# 출력값은 10 크기의 배열의 출력
# 행렬곱 조건을 모두 만족


print()

# 입력 데이터를 여러개로 묶은 것을 배치(batch)라 하며 컴퓨터는 큰 배열을 한 번에 계산하는 것이 더 효율적이므로 배치로 데이터를 묶어 처리하는 것이 바람직

# 이미지 데이터 및 가중치 불러오기
x, t = get_data()
network = init_network()

# 이미지를 묶을 배치 크기를 100으로 지정(100개의 이미지를 하나의 배치로 지정)
batch_size = 100
accuracy_cnt = 0

# 이미지 객체 배열의 크기(10,000)를 batch_size(100) 간격으로 반복
for i in range(0,len(x), batch_size):
    # i번째 인덱스부터 i+배치 크기(100) 직전 인덱스까지의 이미지 객체 배열을 x_batch에 저장 (이미지 객체를 100개 저장)
    x_batch = x[i:i+batch_size]
    # x_batch에 저장된 모든 이미지 정보를 신경망을 거친 후 결과를 y_batch에 저장 (100(이미지 수)×10(0~9까지의 숫자에 해당하는 출력층) 행렬)
    y_batch = predict(network, x_batch)
    # y_batch에서 1번째 차원에 해당하는 값(각 이미지 추론 결과에서 가장 큰 원소(높은 확률)의 인덱스)를 저장(100 크기의 배열)
    p = np.argmax(y_batch, axis=1)
    # 정확도 변수에 추론 결과값 배열(p)과 [i]번째 인덱스 부터 [i+배치크기]번째 인덱스 직전까지의 시험용 이미지 레이블의 배열에서 같은 인덱스의 원소가 일치하는 횟수를 더하여 저장
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
