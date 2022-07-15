# coding: utf-8
import sys
sys.path.append("/Users/hyeon/Documents/Python/DeepLearningStudy")
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import *


# MNIST 데이터셋 불러오기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 총 훈련 크기를 훈련 데이터의 수(x_train 행렬의 0번째 차원의 크기)로 지정
train_size = x_train.shape[0]
# 배치 크기 128
batch_size = 128
# 최대 반복 수 2000
max_iterations = 2000


# 최적화 기법을 저장할 딕셔너리 생성
optimizers = {}
# 'SGD' 최적화 기법 클래스를 딕셔너리에 저장
optimizers['SGD'] = SGD()
# '모멘텀' 최적화 기법 클래스를 딕셔너리에 저장
optimizers['Momentum'] = Momentum()
# 'AdaGrad' 최적화 기법 클래스를 딕셔너리에 저장
optimizers['AdaGrad'] = AdaGrad()
# 'Adam' 최적화 기법 클래스를 딕셔너리에 저장
optimizers['Adam'] = Adam()

# 신경망을 저장할 딕셔너리 생성
networks = {}
# 기존 손실 함수 기록을 저장할 딕셔너리 생성
train_loss = {}
# 최적화 기법을 하나씩 불러와 반복
for key in optimizers.keys():
    # 입력값 784개, 100개의 뉴런으로 구성된 은닉층 4개, 출력값 10개를 가진 신경망을 해당 최적화 기법 전용 신경망으로 지정
    networks[key] = MultiLayerNet(
        input_size=784, hidden_size_list=[100, 100, 100, 100],
        output_size=10)
    # 해당 최적화 기법의 손실 함수를 기록할 리스트 생성
    train_loss[key] = []    


# 최대 반복수만큼 반복
for i in range(max_iterations):
    # 총 학습 데이터 충에서 랜덤으로 배치 데이터 추리기
    batch_mask = np.random.choice(train_size, batch_size)
    # 배치 학습 데이터 저장
    x_batch = x_train[batch_mask]
    # 배치 학습 데이터 정답 레이블 저장
    t_batch = t_train[batch_mask]
    
    # 최적화 기법 별로 반복
    for key in optimizers.keys():
        # 현재 최적화 기법 신경망의 현재 배치 데이터 기울기 계산
        grads = networks[key].gradient(x_batch, t_batch)
        # 현재 최적화 기법 신경망의 매개변수 갱신
        optimizers[key].update(networks[key].params, grads)
        # 현재 최적화 기법 신경망의 현재 배치 데이터 손실값 계산
        loss = networks[key].loss(x_batch, t_batch)
        # 현재 최적화 기법 신경망의 현재 배치 데이터 손실값을 손실값 리스트에 추가
        train_loss[key].append(loss)
    
    # 반복 횟수가 100의 배수이면 아래 내용 실행
    if i % 100 == 0:
        # 현재의 반복 수 출력
        print( "===========" + "iteration:" + str(i) + "===========")
        # 각 최적화 기법 별로 반복
        for key in optimizers.keys():
            # 현재 최적화 기법 신경망의 손실 값 계산
            loss = networks[key].loss(x_batch, t_batch)
            # 현재 최적화 기법의 손실값 출력
            print(key + ":" + str(loss))


# 그래프에서 "SGD"의 점은 o모양, "모멘텀"의 점은 x 모양, "AdaGrad"의 점은 사각형 모양, "Ada"의 점은 마름모 모양으로 지정하기 위한 닥셔너리
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
# x 값에 최대 반복수(2000) 크기의 배열로 저장
x = np.arange(max_iterations)
# 최적화 기법 별로 반복
for key in optimizers.keys():
    # 현재 최적화 기법의 손실 값을 x축으로 하는 그래프 그리기
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()