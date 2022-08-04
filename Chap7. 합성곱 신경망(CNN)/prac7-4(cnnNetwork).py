# 구현한 합성곱 계층과 풀링 계층을 조합하여 MNIST 데이터 손글씨 숫자를 인식하는 CNN 구현
# 구현할 CNN 네트워크는 입력값->합성곱->ReLU->풀링->Affine->ReLU->Affine->Softmax->출력값 구조로 구성
import sys
sys.path.append("/Users/hyeon/Documents/Python/DeepLearningStudy")
from common.layers import *
import numpy as np
from collections import OrderedDict
import pickle

# 상기 구조를 가진 CNN 네트워크 클래스 정의
class SimpleConvNet:
    # 입력 데이터 행렬의 차원(기본 1, 28, 28), 합성곱 매개변수(필터 수, 필터 크기, 패딩, 스트라이드) 딕셔너리, 은닉층 개수, 출력층 개수, 가중치 초기화 표준편차 입력받음
    def __init__(self, input_dim=(1,28,28), conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1}, hidden_size=100, output_size=10, weight_init_std=0.01):
        # 합성곱 매개변수 딕셔너리에서 해당 매개변수를 불러옴
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        # 입력 데이터 행렬의 1차원 값에서 입력 데이터 크기를 불러옴
        input_size = input_dim[1]
        # 합성곱 출력 크기 계산
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        # 풀링 출력 크기 계산
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        # 신경망의 매개변수 저장할 딕셔너리 생성
        self.params = {}
        # 첫번째 은닉층 가중치 : 지정된 표준편차의 표준정규분포 난수를 가진 (필터 개수, 입력 데이터 행렬의 0차원, 필터값 크기, 필터값 크기) 형상의 4차원 행렬
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        # 첫번째 은닉층 편향 : 필터 수 크기의 행렬
        self.params['b1'] = np.zeros(filter_num)
        # 두번째 은닉층 가중치 : 지정된 표준편차의 표준정규분포 난수를 가진 (풀링 출력 크기, 은닉층 크기) 형상의 2차원 행렬
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        # 두번째 은닉층 편향 : 은닉층 크기의 행렬
        self.params['b2'] = np.zeros(hidden_size)
        # 세번째 은닉층 편향 : 지정된 표준편차의 표준정규분포 난수를 가진 (은닉층 크기, 출력층 크기) 형상의 2차원 행렬
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # 세번째 은닉층 편향 : 출력값 크기의 행렬
        self.params['b3'] = np.zeros(output_size)

        # 신경망의 각 계층을 순서가 존재하는 딕셔너리로 지정
        self.layers = OrderedDict()
        # 첫번째 합성곱 계층을 합성곱 클래스에 첫번째 가중치, 첫번째 편향, 스트라이드, 패딩 정보를 입력하여 저장
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad'])
        # 첫번째 ReLU 계층을 ReLU 클래스로 저장
        self.layers['Relu1'] = Relu()
        # 첫번째 풀링 계층을 풀링 클래스에 풀링값 높이(2), 풀링값 너비(2), 스트라이드(2)를 입력하여 저장
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        # 첫번째 Affine(완전연결) 계층을 Affine 클래스에 두번째 가중치, 두번째 편향을 입력하여 저장
        self.layers['Affine1']= Affine(self.params['W2'], self.params['b2'])
        # 두번째 ReLU 계층을 ReLU 클래스로 저장
        self.layers['Relu2'] = Relu()
        # 두번째 Affine(완전연결) 계층을 Affine 클래스에 세번째 가중치, 세번째 편향을 입력하여 저장
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        # 최종 계층을 손실함수(소프트맥스 함수 + 교차 엔트로피 오차 함수) 클래스에 저장
        self.last_layer = SoftmaxWithLoss()

    # 입력값 x를 입력받는 신경망 추론 함수(순전파 계산)
    def predict(self, x):
        # 각 계층별로 반복
        for layer in self.layers.values():
            # 현재 계층 순전파 계산에 x를 입력한 결과를 x에 저장
            x = layer.forward(x)
        # 최종 순전파 계산 결과 x값 리턴
        return x

    # 입력값 x와 정답 레이블 t를 입력받아 손실 값을 구하는 손실함수 (오차 검증)
    def loss(self, x, t):
        # 입력값 x를 순전파 계산한 결과 값(추론 값)을 결과값으로 y에 저장
        y = self.predict(x)
        # 최종 계층(손실 함수) 순전파 계산에 결과 값(y)와 정답 레이블(t)를 입력하여 손실함수 오차 계산 후 리턴
        return self.last_layer.forward(y, t)

    # 입력값 x와 정답 레이블 t를 입력받아 역전파 계산을 하는 기울기 함수 (매개변수 갱신값 추출)
    def gradient(self, x, t):
        # 입력 값과 정답 레이블을 손실 함수에 입력하여 손실값 계산 (순전파)
        self.loss(x,t)

        # 최종 계층(손실 함수) 역전파 계산
        dout = 1
        dout = self.last_layer.backward(dout)

        # 각 계층(클래스)을 리스트로 변환하여 layers로 저장
        layers = list(self.layers.values())
        # 각 계층들의 클래스가 저장된 layers 리스트의 순서를 반대로 지정
        layers.reverse()
        # layers 리스트의 각 클래스 반복(순전파 계산 순서의 역방향으로 반복)
        for layer in layers:
            # 각 계층 클래스의 역전파 계산
            dout = layer.backward(dout)

        # 기울기 값 저장할 딕셔너리 생성
        grads = {}
        # 첫번째 은닉층 가중치의 기울기 (Conv1 계층에서의 가중치 기울기)
        grads['W1'] = self.layers['Conv1'].dW
        # 첫번쨰 은닉층 편향의 기울기 (Conv1 계층에서의 편향 기울기)
        grads['b1'] = self.layers['Conv1'].db
        # 두번째 은닉층 가중치의 기울기 (Affine1 계층에서의 가중치 기울기)
        grads['W2'] = self.layers['Affine1'].dW
        # 두번째 은닉층 편향의 기울기 (Affine1 계층에서의 편향 기울기)
        grads['b2'] = self.layers['Affine1'].db
        # 세번째 은닉층 가중치의 기울기 (Affine2 계층에서의 가중치 기울기)
        grads['W3'] = self.layers['Affine2'].dW
        # 세번째 은닉층 가중치의 기울기 (Affine2 계층에서의 편향 기울기)
        grads['b3'] = self.layers['Affine2'].db

        # 모든 매개변수 기울기가 저장된 딕셔너리 리턴
        return grads


    # 정확도 계산하는 함수
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

    # 매개변수를 피클 파일로 저장하는 함수
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    # 매개변수 피클 파일을 로드하는 함수
    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]



# 구현한 CNN 네트워크를 이용하여 MNIST 데이터셋 학습시키기

import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.trainer import Trainer

# MNIST 데이터셋 로드
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 훈련 데이터, 정답 레이블을 5000개로 제한 (학습 시간 절감을 위해)
x_train, t_train = x_train[:5000], t_train[:5000]
# 시험 데이터, 정답 레이블을 1000개로 제한
x_test, t_test = x_test[:1000], t_test[:1000]

# 최대 에폭 20
max_epochs = 20

# 합성곱 신경망 클래스 생성
network = SimpleConvNet(input_dim=(1,28,28), conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1}, hidden_size=100, output_size=10, weight_init_std=0.01)
# 신경망 학습을 담당하는 학습 클래스 생성                     
trainer = Trainer(network, x_train, t_train, x_test, t_test, epochs=max_epochs, mini_batch_size=100, optimizer='Adam', optimizer_param={'lr': 0.001}, evaluate_sample_num_per_epoch=1000)
# 신경망 학습
trainer.train()

# 신경망 매개변수를 피클파일로 저장
network.save_params("params.pkl")
# 피클 파일 저장시 문자열 출력
print("Saved Network Parameters!")

# 학습 데이터와 시험 데이터 추론 정확도를 그래프로 시각화
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()




# 신경망이 학습할 때 사용하는 필터값을 시각화하여 확인
# 1번째 합성곱 계층의 가중치는 형상이 (30,1,5,5)로, 배치 크기(30), 채널 수(1), 세로 크기(5), 가로 크기(5)를 의미
# 이 가중치(필터) 값을 5×5 크기의 1채널 흑백 이미지로 시각화 가능

import numpy as np
import matplotlib.pyplot as plt

# 필터값 시각화하는 함수
def filter_show(filters, nx=8, margin=3, scale=10):
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()

# 신경망 불러와서 (학습되지 않은) 첫번째 가중치(필터값) 시각화
network = SimpleConvNet()
filter_show(network.params['W1'])

# 학습된 첫번째 가중치(필터값)을 피클파일에서 불러와 시각화
network.load_params("params.pkl")
filter_show(network.params['W1'])


# 학습되지 않은 상태의 필터값을 시각화하면 규칙성이 없는 무작위 흑백 이미지가 나오지만
# 학습된 상태의 필터값을 시각화하면 상대적으로 규직성이 있는 흑백 이미지가 출력됨
# 학습된 상태의 필터값은 이미지 상에서 에지(경계선), 블롭(덩어리) 등의 요소에 주목하고 있다는 뜻
# 이러한 필터(매개변수) 은닉층을 깊게 하면 깊은 층의 필터값일수록 이미지 상에서 더욱 추상화된 정보(텍스쳐, 사물)를 구별하게 됨