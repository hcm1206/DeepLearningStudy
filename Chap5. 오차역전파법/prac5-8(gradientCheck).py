# 수치 미분으로 기울기 구하기 : 상대적으로 구현은 쉽지만 계산 속도가 느림
# 오차역전파법(해석적으로 수식을 풀어 계산)으로 기울기 구하기 : 상대적으로 구현이 어렵지만 계산 속도가 빠름

# 일반적으로 오차역전파법을 많이 사용하고 앞으로 오차역전파법을 사용할 예정
# 그러나 오차역전파법은 구현이 복잡하여 제대로 구현했는지 확인하는 과정이 필요
# 수치 미분을 이용한 기울기 연산과 오차역전파법을 이용한 기울기 연산을 비교하여 기울기 구현이 정확한지 확인
# 이러한 작업을 기울기 확인(gradient check)이라고 함

import numpy as np
import sys
sys.path.append("/Users/hyeon/Documents/Python/DeepLearningStudy")
# mnist 데이터셋 불러오기
from dataset.mnist import load_mnist
# 구현한 2층 신경망 클래스 불러오기
from TwoLayerNet import TwoLayerNet

# mnist 데이터 셋 불러와 변수에 각각 저장
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# 2층 신경망을 불러와 network에 저장
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 훈련 데이터 3개(0~2 인덱스)를 불러와 x_batch에 배치로 저장
x_batch = x_train[:3]
# 정답 데이터 3개(0~2 인덱스)를 불러와 t_batch에 배치로 저장 
t_batch = t_train[:3]

# 배치 데이터를 신경망의 수치 미분을 통해 기울기 구하기
grad_numerical = network.numerical_gradient(x_batch, t_batch)
# 배치 데이터를 신경망의 오차역전파법을 통해 기울기 구하기
grad_backprop = network.gradient(x_batch, t_batch)

# 수치 미분을 통해 구한 기울기(딕셔너리)의 키(매개변수) 반복
for key in grad_numerical.keys():
    # 역전파법을 통해 구한 해당 매개변수 기울기와 수치미분을 통해 구한 해당 매개변수 기울기의 차를 구하여 절대값을 취한 배열 원소들의 평균값을 diff에 저장
    # 수치 미분으로 구한 기울기와 오차역전파법으로 구한 기울기의 평균 차를 구하는 것
    diff = np.average(np.abs(grad_backprop[key]-grad_numerical[key]))
    # diff(기울기의 평균 차) 출력
    print(key + ":" + str(diff))

    # e-x가 뒤에 붙어있으면 10의 -x제곱이라는 뜻으로 아주 작은 값을 의미