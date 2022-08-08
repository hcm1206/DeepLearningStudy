# 일반적으로 딥러닝 학습은 굉장히 오랜 시간이 소요 : 예를 들어 이미지 인식을 위해 순전파 처리를 할 때 합성곱 계산에 많은 시간 소요
# 딥러닝 고속화에는 GPU(+최적화 라이브러리) 사용, 분산 학습 등이 존재
# 이 외에도 연산 정밀도와 비트를 줄이는 방식으로도 고속화 가능
# 연산 정밀도, 비트를 줄인다는 것은 계산할 데이터의 비트 수를 줄여 메모리(RAM) 용량을 확보하고 버스 대역폭 면에서 병목현상을 줄여 성능을 향상시키는 것을 의미

# 컴퓨터에서 보통 64비트 또는 32비트 부동소수점(float) 수를 이용하여 실수 표현
# 사용 비트 수가 많아질수록 계산(특히 소수 계산)이 정밀해지는 효과를 얻을 수 있으나 그만큼 계산 시간과 메모리 사용량이 증가
# 그런데 딥러닝 학습에서는 계산이 정밀할 필요가 없기 때문에 높은 수치 정밀도(실수 표현에 사용하는 비트)가 높을 필요가 없음
# 따라서 계산시 사용하는 비트 수를 줄여 딥러닝 학습 시간 단축 가능

# 비트 수를 줄여도 계산 결과에 실제로 영향이 없는지 테스트
# 64비트 부동 소수점과 16비트 부동 소수점을 이용하여 각각 딥러닝 학습을 진행한 후 정확도 비교

import sys
sys.path.append("/Users/hyeon/Documents/Python/DeepLearningStudy")
import numpy as np
import matplotlib.pyplot as plt
from deep_convnet import DeepConvNet
from dataset.mnist import load_mnist

# MNIST 데이터셋 로드
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 사용할 신경망은 이전에 구현했던 심층 신경망 클래스 사용
network = DeepConvNet()
# 미리 학습된 매개변수를 피클 파일에서 불러와 적용
network.load_params("deep_convnet_params.pkl")

# 샘플 데이터 수 10000개로 지정(연산 시간이 오래 걸릴 시 이 변수의 숫자를 줄여서 테스트)
sampled = 10000 
# 시험 데이터를 샘플 데이터 수로 제한(10000개)
x_test = x_test[:sampled]
t_test = t_test[:sampled]

# 일반적인 설정(64비트 부동소수점)일 때 신경망 추론 정확도 측정
print("caluculate accuracy (float64) ... ")
print(network.accuracy(x_test, t_test))

# 시험 데이터를 16비트 부동소수점 데이터로 변환
x_test = x_test.astype(np.float16)
# 신경망의 모든 매개변수 반복
for param in network.params.values():
    # 해당 매개변수의 모든 데이터를 16비트 부동소수점 데이터로 변환
    param[...] = param.astype(np.float16)

# 16비트 부동소수점일 때 신경망 추론 정확도 측정
print("caluculate accuracy (float16) ... ")
print(network.accuracy(x_test, t_test))


# 측정 결과 64비트 부동소수점 데이터를 사용할 때와 16비트 부동소수점 데이터를 사용할 때의 추론 정확도가 일치
# 딥러닝 학습에서는 데이터가 정밀하지 않아도 일관적인 정확도를 보인다는 의미
# 따라서 데이터 학습 시에는 계산 시간 단축과 컴퓨팅 자원 사용의 효율성을 위해 연산 정밀도를 줄여주는 것도 좋음