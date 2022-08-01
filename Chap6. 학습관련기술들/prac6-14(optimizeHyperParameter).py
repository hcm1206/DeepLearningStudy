# 하이퍼파라미터를 최적화 할 때는 최적값이 포함된 범위를 조금씩 줄여나가며 그 범위 내의 최적값을 무작위로 선정하여 테스트 후 다시 범위를 줄여나가는 과정을 거침
# 결론적으로 하이퍼파라미터 최적화는 최적값의 범위를 좁혀가는 것이 핵심
# 이 때 최적값의 범위에서 최적값을 규칙적으로 선별하지 않고 무작위로 선별해야 더 좋은 결과를 낸다고 함

# 하이퍼파라미터 범위는 0.001 ~ 1000과 같이 10의 거듭제곱 단위로 대략적으로 지정하는 것이 효과적
# 이러한 10의 거듭제곱 범위를 로그 스케일(log scale)이라 함

# 범위 지정 후 하이퍼파라미터값을 무작위로 추출한 뒤 이 하이퍼파라미터값으로 학습을 진행하여 정확도를 평가 한 후 하이퍼파라미터 범위 값을 갱신하는 과정 반복


# MNIST 데이터셋으로 하이퍼파라미터 최적화해보기
# 최적화할 하이퍼파라미터 값은 학습률, 가중치 감소 계수 2가지

# 하이퍼파라미터 값의 검증은 로그 스케일(10의 거듭제곱) 범위에서 무작위 추출하여 수행
# 가중치 감소 계수의 최초 범위 : 10^-8 ~ 10^-4
# 학습률의 최초 범위 : 10^-6 ~ 10^-2
# 로그 스케일 범위 무작위 추출은 10 ** np.random.uniform(x1, x2) 식으로 나타낼 수 있음 (10^x1 ~ 10^x2)

import sys
sys.path.append("/Users/hyeon/Documents/Python/DeepLearningStudy")
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.util import shuffle_dataset
from common.trainer import Trainer

# MNIST 데이터셋 로드
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 학습 데이터 수를 500개로 설정
x_train = x_train[:500]
t_train = t_train[:500]

# 검증 데이터 비율을 학습 데이터의 0.2로 설정
validation_rate = 0.20
# 검증 데이터의 수 지정 (500 * 0.2 == 100)
validation_num = int(x_train.shape[0] * validation_rate)
# 검증 데이터를 무작위로 추출하기 위해 학습 데이터를 섞음
x_train, t_train = shuffle_dataset(x_train, t_train)
# 기존 학습 데이터 중에서 100개를 검증 데이터로 지정
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
# 기존 학습 데이터 중에서 나머지 400개를 학습 데이터로 지정
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]

# 학습률과 가중치 감소 계수, 에폭(기본값 50)을 입력받아 신경망 학습을 담당하는 클래스 정의
def __train(lr, weight_decay, epocs=50):
    # 신경망 클래스 생성 (가중치 감소 계수를 weight_decay로 입력받음)
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10, weight_decay_lambda=weight_decay)
    # 학습 클래스 생성 (최적화시 학습율을 lr로 입력받음)
    trainer = Trainer(network, x_train, t_train, x_val, t_val, epochs=epocs, mini_batch_size=100, optimizer='sgd', optimizer_param={'lr': lr}, verbose=False)
    # 학습 클래스를 통해 입력한 조건에 따라 신경망 학습
    trainer.train()

    # 학습된 신경망의 시험 데이터 추론 정확도, 학습 데이터 추론 정확도 리턴
    return trainer.test_acc_list, trainer.train_acc_list

# 하이퍼파라미터 검증 횟수를 100으로 설정
optimization_trial = 100
# 검증 데이터 추론 정확도를 저장할 딕셔너리 생성
results_val = {}
# 학습 데이터 추론 정확도를 저장할 딕셔너리 생성
results_train = {}
# 하이퍼파라미터 검증 횟수(100)만큼 반복
for _ in range(optimization_trial):
    # 가중치 감소 계수를 10**-8 ~ 10**-4 사이 로그 스케일 범위에서 무작위로 추출하여 지정
    weight_decay = 10 ** np.random.uniform(-8, -4)
    # 학습률을 10**-6 ~ 10**-2 사이 로그 스케일 범위에서 무작위로 추출하여 지정
    lr = 10 ** np.random.uniform(-6, -2)

    # 학습률과 가중치 감소 계수를 적용하여 학습을 진행한 후 검증 데이터 정확도와 학습 데이터 정확도를 받아옴
    val_acc_list, train_acc_list = __train(lr, weight_decay)
    # (가장 마지막 인덱스의) 검증 데이터 추론 정확도, 가중치 감수 계수 출력
    print("val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay))
    # 현재 하이퍼파라미터 검증에서의 학습률과 가중치 감소율(하이퍼파라미터 상태)을 문자열로 key에 저장
    key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
    # 현재 하이퍼파라미터 상태의 검증 데이터 추론 정확도를 딕셔너리로 저장
    results_val[key] = val_acc_list
    # 현재 하이퍼파라미터 상태의 학습 데이터 추론 정확도를 딕셔너리로 저장
    results_train[key] = train_acc_list


# 지정한 범위에서의 하이퍼파라미터 검증 횟수를 모두 마친후 결과 출력
print("=========== Hyper-Parameter Optimization Result ===========")
# 출력할 그래프 개수 20
graph_draw_num = 20
# 열 개수 5
col_num = 5
# 행 개수를 그래프 개수를 열 개수로 나눈 값에서 가장 가까운 정수로 지정 (4)
row_num = int(np.ceil(graph_draw_num / col_num))
# 그래프 출력 횟수를 저장할 변수
i = 0

# 검증 데이터 추론 정확도를 내림차순으로 정렬한 딕셔너리를 반복 (인덱스가 작을수록 추론 정확도가 높음)
for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
    # 현재 그래프는 i+1번째로 가장 정확한 그래프이고, 검증 데이터 추론 정확도 출력
    print("Best-" + str(i+1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)

    # 설정한 행의 수와 열의 수 대로 그래프를 배열
    plt.subplot(row_num, col_num, i+1)
    # 현재 그래프 제목에서 i+1번쨰로 정확한 그래프라고 표시
    plt.title("Best-" + str(i+1))
    # y값 범위를 0.0~1.0으로 지정
    plt.ylim(0.0, 1.0)
    # 반복 횟수가 5의 배수(0,5,10,15,20)라면 y축 정보 표시
    if i % 5: plt.yticks([])
    # x축 정보 표시
    plt.xticks([])
    # 입력값은 검증 데이터 추론 정확도 리스트의 크기만큼의 배열
    x = np.arange(len(val_acc_list))
    # 입력값에 따른 검증 데이터 추론 정확도 표시 
    plt.plot(x, val_acc_list)
    # 입력값에 따른 학습 데이터 추론 정확도 표시
    plt.plot(x, results_train[key], "--")
    # 반복 횟수 1 증가
    i += 1

    # 반복 횟수가 그래프 출력 횟수보다 높으면(20번째 그래프가 그려졌다면)
    if i >= graph_draw_num:
        # 반복 종료
        break

plt.show()


# 실행하면 100번의 반복을 진행하여 학습률과 가중치 감수 계수에 따른 추론 정확도를 출력
# 최종적으로 100번의 반복 중 정확도가 가장 높은 상위 20개의 경우를 그래프로 나타냄

# 정확도가 높게 나온 케이스의 학습률과 가중치 감소 계수를 보면
# 학습률은 10**-3 ~ 10**-2 범위에 분포
# 가중치 감소 계수는 10**-5 ~ 10**-8 ~ 10**-5 사이에 분포

# 이렇게 상위 케이스를 분석하여 하이퍼파라미터의 분포값을 좁힌 후 다시 테스트하는 과정을 반복하여 최적의 하이퍼파라미터 값을 결정