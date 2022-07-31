# 가중치 감소는 간단하지만 신경망 대응이 복잡해지면 가중치 감소만으로 대응하기 어려워짐
# 오버피팅을 억제하는 방법으로 드롭 아웃도 많이 사용
# 드롭 아웃(Dropout)은 훈련 때 은닉층의 무작위 뉴런을 삭제하여 학습하는 방법
# 훈련 때는 무작위의 뉴런을 삭제하여 신호를 흘리지만 시험(실전 사용) 때에는 모든 뉴런을 사용하여 신호 전달
# 시험 때는 각 뉴런의 출력에 훈련 시 삭제하지 않은 비율을 곱하여 출력


# 드롭아웃 구현 예시

import numpy as np

class Dropout:
    #  드롭아웃 클래스 생성 시 드롭아웃 비율을 입력받음(기본 0.5)
    def __init__(self, dropout_ratio = 0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    # 순전파 계산(입력값과 드롭아웃 실행 여부(train_flg) 입력받음)
    def forward(self, x, train_flg=True):
        # 드롭아웃을 실행한다면
        if train_flg:
            # 입력값 배열과 같은 형상의 랜덤 범위(0~1)값이 저장된 배열을 생성하여 드롭아웃 비율보다 높은 값의 인덱스 원소만 True로 설정
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            # 입력값에서 True인 인덱스의 값만 출력
            return x * self.mask
        # 드롭아웃을 실행하지 않는다면
        else:
            # 입력값 배열에 단순히 0.5(1-드롭아웃 비율(0.5))를 곱함
            return x * (1.0-self.dropout_ratio)

    # 역전파 계산
    def backward(self, dout):
        # 순전파에서 True로 지정된 인덱스의 값만 신호 전달
        return dout * self.mask

# 순전파 계산시 입력값 행렬의 각 원소마다 0~1 사이의 랜덤 값을 할당하여 그 값이 0.5 이상일 때만 활성화하고 0.5 미만이면 비활성화하는 방식으로 드롭아웃 구현
# 역전파 계산시에는 순전파때 통과된 원소값만 통과시키고 순전파에서 비활성화된 원소값은 마찬가지로 차단



# MNIST 데이터셋으로 드롭아웃 효과 테스트
import sys
sys.path.append("/Users/hyeon/Documents/Python/DeepLearningStudy")
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
# 신경망 학습 전용 클래스를 생성하여 학습을 Trainer 클래스 상에서 이루어지도록 함
from common.trainer import Trainer

# MNIST 신경망 불러오기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 오버피팅 발생시키기 위해 훈련 데이터 수를 300개로 제한
x_train = x_train[:300]
t_train = t_train[:300]

# 드롭아웃 적용 여부 변수 생성(True)
use_dropout = True
# 드롭아웃 비율을 0.2로 설정
dropout_ratio = 0.2

# 드롭아웃을 활성화하고 드롭아웃 비율을 0.2로 설정한 신경망 생성
network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10, use_dropout=use_dropout, dropout_ration=dropout_ratio)
# 신경망 학습 클래스 생성 (신경망, 훈련 데이터 및 시험 데이터, 에폭, 미니 배치 크기, 매개변수 최적화 기법, 매개변수 최적화시 학습률, 훈련시 훈련 과정(추론 정확도 등)을 출력할지 여부(verbose) 입력)
trainer = Trainer(network, x_train, t_train, x_test, t_test, epochs=301, mini_batch_size=100, optimizer='sgd', optimizer_param={'lr': 0.01}, verbose=True)
# 학습 클래스 상에서 신경망 학습 진행
trainer.train()

# 학습이 완료된 학습 클래스에서 학습 데이터 추론 정확도 리스트와 시험 데이터 추론 정확도 리스트 불러오기
train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

# 학습 데이터용 그래프 마커를 원형, 시험 데이터용 그래프 마커를 사각형으로 지정
markers = {'train': 'o', 'test': 's'}
# 학습 데이터 정확도 리스트의 원소 수만큼 입력값의 범위를 배열로 지정
x = np.arange(len(train_acc_list))
# 입력값 x에 대응하는 학습 데이터 추론 정확도를 그래프로 표시
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
# 입력값 x에 대응하는 시험 데이터 추론 정확도를 그래프로 표시
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()


# 테스트 결과 드롭아웃 사용 시 훈련 데이터와 시험 데이터 간의 추론 정확도 차이가 드롭아웃을 사용하지 않았을 때보다 상대적으로 적음
# 훈련 데이터의 추론 정확도도 다소 낮아지는 현상 발생