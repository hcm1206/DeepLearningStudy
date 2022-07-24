# 배치 정규화(Batch Normalization) : 각 은닉층에서의 활성화 값이 적절하게 분포되도록 임의로 활성화 값을 조정하는 것
# 기존 신경망에서 Affine 계층과 활성화 함수 계층 사이에 '배치 정규화' 계층을 집어 넣어 계산된 값을 활성화 함수에 입력하기 전 정규화를 통해 분포시킴
# Affine 계산 데이터를 평균이 0, 분산이 1이 되도록 정규화

# MNIST 신경망 학습에서 배치 정규화 계층을 사용하여 비교
# 다양한 비교를 위해 16개의 다른 초기 가중치 값을 부여하여 16가지 학습 비교

import sys
sys.path.append("/Users/hyeon/Documents/Python/DeepLearningStudy")
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam

# 훈련 데이터 불러오기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 학습 데이터 & 시험 데이터를 1000개로 제한
x_train = x_train[:1000]
t_train = t_train[:1000]

# 최대 에폭 수 20
max_epochs = 20
# 훈련량 : 훈련 데이터 행렬의 0차원 크기 (1000)
train_size = x_train.shape[0]
# 배치 크기 100
batch_size = 100
# 학습률 0.01
learning_rate = 0.01

# 초기 가중치를 입력받는 훈련 클래스 정의
def __train(weight_init_std):
    # 배치 정규화를 진행하는 신경망 클래스 생성 (use_batchnorm = True)
    bn_network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10, weight_init_std=weight_init_std, use_batchnorm=True)
    # 배치 정규화를 진행하지 않는 신경망 클래스 생성
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10, weight_init_std=weight_init_std)
    # 매개변수 최적화는 SGD(확률적 경사 하강법) 사용
    optimizer = SGD(lr=learning_rate)
    
    # 배치 정규화하지 않은 신경망의 정확도를 기록할 리스트 생성
    train_acc_list = []
    # 배차 정규화한 신경망의 정확도를 기록할 리스트 생성
    bn_train_acc_list = []
    
    # 에폭별 반복 횟수 : 훈련 크기를 배치 크기로 나눈 값 또는 1
    iter_per_epoch = max(train_size / batch_size, 1)
    # 에폭 횟수 계산할 변수
    epoch_cnt = 0
    
    # 1,000,000,000(10억)번 반복
    for i in range(1000000000):
        # 배치로 사용할 데이터 인덱스 선정
        batch_mask = np.random.choice(train_size, batch_size)
        # 훈련용 배치 데이터 저장
        x_batch = x_train[batch_mask]
        # 시험용 배치 데이터 저장
        t_batch = t_train[batch_mask]

        # 정규화된 신경망과 정규화되지 않은 신경망을 차례로 반복
        for _network in (bn_network, network):
            # 신경망의 훈련 데이터와 시험 데이터를 입력하여 기울기 계산
            grads = _network.gradient(x_batch, t_batch)
            # 신경망 매개변수 갱신
            optimizer.update(_network.params, grads)

        # 반복 횟수가 에폭별 반복 횟수의 배수일 때
        if i % iter_per_epoch == 0:
            # 배치 정규화하지 않은 신경망의 추론 정확도 계산
            train_acc = network.accuracy(x_train, t_train)
            # 배치 정규화한 신경망의 추론 정확도 계산
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            # 현재 배치 정규화하지 않은 신경망의 추론 정확도를 리스트에 저장
            train_acc_list.append(train_acc)
            # 현재 배치 정규화한 신경망의 추론 정확도를 리스트에 저장
            bn_train_acc_list.append(bn_train_acc)
            # 현재 에폭 횟수에서의 각 신경망의 추론 정확도 출력
            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))
            # 에폭 횟수 1 증가
            epoch_cnt += 1
            # 에폭 횟수가 최대 에폭 횟수에 도달했다면
            if epoch_cnt >= max_epochs:
                # 학습 종료
                break
    # 배치 정규화하지 않은 신경망과 배치 정규화한 신경망의 정확도 리스트를 각각 리턴          
    return train_acc_list, bn_train_acc_list


# 0부터 -4까지 16개의 로그 스케일값 배열 생성
weight_scale_list = np.logspace(0, -4, num=16)
# x에 0부터 최대 에폭수(20) 만큼의 배열 저장
x = np.arange(max_epochs)

# 로그 스케일값 배열을 인덱스와 변수값대로 반복
for i, w in enumerate(weight_scale_list):
    # 현재 반복 횟수 출력
    print( "============== " + str(i+1) + "/16" + " ==============")
    # 초기 가중치를 입력받은 훈련 클래스를 통해 배치 정규화되지 않은 신경망 추론 정확도와 배치 정규화된 신경망 추론 정확도 저장
    train_acc_list, bn_train_acc_list = __train(w)
    # 4행 4열의 그래프들 중 i+1번째 그래프 설정
    plt.subplot(4,4,i+1)
    # 현재 그래프의 초기 가중치값을 그래프 제목으로 설정
    plt.title("W:" + str(w))
    # 현재 그래프가 마지막 그래프이면
    if i == 15:
        # 그래프 범례 표기하여 그래프 그리기 (x축은 에폭, y축은 에폭에 따른 신경망 추론 정확도)
        plt.plot(x, bn_train_acc_list, label='Batch Normalization', markevery=2)
        plt.plot(x, train_acc_list, linestyle = "--", label='Normal(without BatchNorm)', markevery=2)
    # 현재 그래프가 마지막 그래프가 아니라면
    else:
        # 그래프 범례를 표기하지 않고 그래프 그리기
        plt.plot(x, bn_train_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle="--", markevery=2)
    # y값의 범위를 0부터 1.0까지로 설정
    plt.ylim(0, 1.0)
    # 특정 그래프가 아니면
    if i % 4:
        # y축에 아무것도 표기하지 않음
        plt.yticks([])
    # 현재 그래프가 1, 5, 9, 13번째 그래프라면
    else:
        # y축에 "accuracy" 표기
        plt.ylabel("accuracy")
    # 현재 그래프가 11번째 이하 그래프라면
    if i < 12:
        # x축에 아무것도 표기하지 않음
        plt.xticks([])
    # 현재 그래프가 12번째 이상 그래프라면
    else:
        # x축에 "epochs" 표기
        plt.xlabel("epochs")
    # 범례를 우하단에 위치시킴
    plt.legend(loc='lower right')

# 그래프 그리기
plt.show()


# 그래프에서 실선은 배치 정규화를 사용한 신경망, 점선은 배치 정규화를 사용하지 않은 신경망을 의미
# 정확도가 1에 가까울수록 정확하다는 뜻 (손실 함수 값이 아니라 정확도이므로 0보다 1에 가까워야 더 정확한 값이라는 뜻)

# 대다수의 경우에서 배치 정규화를 사용했을 때 학습 진도가 더 빠르고 배치 정규화를 사용하지 않았을 때 학습이 거의 진행되지 않기도 함