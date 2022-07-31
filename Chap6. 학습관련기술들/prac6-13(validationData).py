# 하이퍼파라미터는 가중치, 편향과 같이 컴퓨터가 스스로 찾아내는 값이 아닌 뉴런 수, 배치 크기, 학습률 등의 사용자가 직접 설정하는 값을 의미
# 딥러닝 신경망에서 하이퍼파라미터를 설정하는 것은 많은 시행착오가 따르는 어려운 일
# 그러나 하이퍼파라미터 값을 제대로 설정하지 않으면 학습 효율이 크게 저하되거나 신경망이 정상동작하지 않는 등의 문제 발생

# 지금까지 신경망 학습 과정에서 학습용 데이터셋은 학습 데이터와 시험 데이터 두 가지로 나누어 사용
# 그러나 하이퍼파라미터의 성능을 평가할 때는 시험 데이터를 사용하면 안 됨 : 시험 데이터에 오버피팅되는 현상이 발생하기 때문
# 따라서 하이퍼파라미터 설정 시에는 하이퍼파라미터 전용 데이터셋이 필요하며 이를 검증 데이터(validation data)라 부름


# MMIST 데이터셋의 경우에는 검증 데이터를 따로 제공하지 않으므로 훈련 데이터에서 일부 데이터를 분리하여 사용

# MNIST 데이터셋의 훈련 데이터에서 검증 데이터를 분리하는 과정 예시

import numpy as np
# 먼저 훈련용 데이터를 랜덤으로 섞어주는 함수 생성(학습 데이터와 정답 레이블 입력받음)
def shuffle_dataset(x, t):
    # 학습 데이터의 수(MNIST의 경우 60,000개) 만큼의 무작위 섞인 배열 생성 (0~59999 범위의 수가 무작위로 섞인 60,000개 원소가 저장된 배열)
    permutation = np.random.permutation(x.shape[0])
    # 학습 데이터의 행렬이 2개 차원이면 0차원(데이터 인덱스)를 무작위로 섞음 {사실 무슨 소린지 모르겠지만 어쨌든 학습 데이터를 랜덤으로 섞는다는 뜻일듯}
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    # 정답 데이터의 행렬을 무작위 인덱스로 섞음
    t = t[permutation]
    # 셔플한 데이터 리턴
    return x, t

import sys
sys.path.append("/Users/hyeon/Documents/Python/DeepLearningStudy")
from dataset.mnist import load_mnist

# MNIST 데이터셋 불러오기
(x_train, t_train), (x_test, t_test) = load_mnist()
# 불러온 학습 데이터와 정답 레이블을 무작위로 섞음
x_train, t_train = shuffle_dataset(x_train, t_train)
# 검증 데이터 비율을 0.2로 설정 (학습 데이터 중 0.2 비율을 검증 데이터로 사용)
validation_rate = 0.20
# 검증 데이터 인덱스 숫자를 학습 데이터 수에서 검증 데이터 비율을 곱한 정수값으로 지정 (60,000 * 0.2 == 12,000)
validation_num = int(x_train.shape[0] * validation_rate)

# (셔플된) 기존 학습 데이터(와 정답 레이블)의 12,000번째 인덱스 직전까지의 데이터를 검증 데이터로 사용
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
# (셔플된) 기존 학습 데이터(와 정답 레이블)의 12,000번째 인덱스부터의 데이터를 학습 데이터로 사용
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]
