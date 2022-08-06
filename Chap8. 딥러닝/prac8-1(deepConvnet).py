# 계층을 더욱 깊게 만든 심층 CNN 신경망 구현
# 이거 일일이 설명하는 것은 엄청난 삽질이기 때문에 대략적으로 어떻게 진행되는지만 보고 넘어감
# 솔직히 상당수 내용 이해 불가


import sys
sys.path.append("/Users/hyeon/Documents/Python/DeepLearningStudy")
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *


# 신경망의 계층 순서
# conv - relu - conv- relu - pool - conv - relu - conv - relu - pool - conv - relu - conv- relu - pool - affine - relu - dropout - affine - dropout - softmax
# 3×3 크기의 필터값을 사용하여 합성곱 계산
# 활성화 함수로 ReLU 사용
# 완전연결(Affine) 계층 뒤에 드롭아웃 계층 사용
# 최적화 기법으로 Adam 사용
# 가중치 초깃값으로 He 초깃값 사용

# 심층 CNN 신경망 클래스 구현
class DeepConvNet:

    # 클래스 생성시 입력값 행렬의 차원,
    def __init__(self, input_dim=(1, 28, 28),
        # 각 합성곱 계층에서 사용하는 매개변수(각 필터(합성곱 매개변수) 숫자, 필터 크기, 패딩, 스트라이드),
        conv_param_1 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
        conv_param_2 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
        conv_param_3 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
        conv_param_4 = {'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1},
        conv_param_5 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
        conv_param_6 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
        # 은닉층 크기, 출력층 크기 입력받음
        hidden_size=50, output_size=10):


        pre_node_nums = np.array([1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, hidden_size])
        weight_init_scales = np.sqrt(2.0 / pre_node_nums)  
        
        self.params = {}
        pre_channel_num = input_dim[0]
        # 각 합성곱 계층마다 인덱스, 매개변수를 불러와 반복
        for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6]):
            # 현재 합성곱 계층 가중치(필터값) 계산
            self.params['W' + str(idx+1)] = weight_init_scales[idx] * np.random.randn(conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])
            # 현재 합성곱 계층의 편향 계산
            self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])
            # 이전 채널 수 불러옴
            pre_channel_num = conv_param['filter_num']
        self.params['W7'] = weight_init_scales[6] * np.random.randn(64*4*4, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = weight_init_scales[7] * np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size)


        # 계층을 저장하는 리스트에 각 계층 클래스 저장
        self.layers = []
        self.layers.append(Convolution(self.params['W1'], self.params['b1'], conv_param_1['stride'], conv_param_1['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W2'], self.params['b2'], conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W3'], self.params['b3'], conv_param_3['stride'], conv_param_3['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W4'], self.params['b4'], conv_param_4['stride'], conv_param_4['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W5'], self.params['b5'], conv_param_5['stride'], conv_param_5['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W6'], self.params['b6'], conv_param_6['stride'], conv_param_6['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Affine(self.params['W7'], self.params['b7']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))
        self.layers.append(Affine(self.params['W8'], self.params['b8']))
        self.layers.append(Dropout(0.5))
        # 최종 계층(소프트맥스 함수 + 손실 함수)은 따로 저장
        self.last_layer = SoftmaxWithLoss()

    # 순전파 계산(드롭아웃 적용)
    def predict(self, x, train_flg=False):
        for layer in self.layers:
            # 드롭아웃 계층이면
            if isinstance(layer, Dropout):
                # 지정된 매개변수만 순전파 계산
                x = layer.forward(x, train_flg)
            # 드롭아웃 계층이 아니면
            else:
                # 모든 매개변수 순전파 계산
                x = layer.forward(x)
        # 최종 순전파 계산 결과 리턴
        return x

    # 손실 함수 계산
    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    # 정확도 계산
    def accuracy(self, x, t, batch_size=100):
        # 배치 데이터용 데이터 변환
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    # 역전파 계산(기울기 게산)
    def gradient(self, x, t):

        self.loss(x, t)


        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 합성곱 계층의 매개변수 기울기 저장할 딕셔너리
        grads = {}
        # 합성곱 계층만 뽑아서 매개변수와 편향 기울기 저장
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW
            grads['b' + str(i+1)] = self.layers[layer_idx].db
        # 합성곱 계층 매개변수가 저장된 딕셔너리 리턴
        return grads

    # 매개변수를 피클 파일로 저장
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    # 피클 파일로 저장된 매개변수 로드
    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            self.layers[layer_idx].W = self.params['W' + str(i+1)]
            self.layers[layer_idx].b = self.params['b' + str(i+1)]



# 신경망 학습시키기

import numpy as np
from dataset.mnist import load_mnist
from common.trainer import Trainer

# MNIST 데이터셋 로드
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 신경망 클래스 불러오기
network = DeepConvNet()  
# 학습 클래스 불러오기
trainer = Trainer(network, x_train, t_train, x_test, t_test, epochs=20, mini_batch_size=100, optimizer='Adam', optimizer_param={'lr':0.001}, evaluate_sample_num_per_epoch=1000)
# 학습 클래스를 이용하여 신경망 학습 (엄청나게 오래걸림)
trainer.train()

# 학습된 신경망 매개변수를 피클 파일로 저장
network.save_params("deep_convnet_params.pkl")
print("Saved Network Parameters!")



# 학습된 데이터 중에서 인식에 실패한 데이터셋의 이미지를 출력해보기

import matplotlib.pyplot as plt
from dataset.mnist import load_mnist


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()
network.load_params("deep_convnet_params.pkl")
print("calculating test accuracy ... ")


classified_ids = []

acc = 0.0
batch_size = 100

for i in range(int(x_test.shape[0] / batch_size)):
    tx = x_test[i*batch_size:(i+1)*batch_size]
    tt = t_test[i*batch_size:(i+1)*batch_size]
    y = network.predict(tx, train_flg=False)
    y = np.argmax(y, axis=1)
    classified_ids.append(y)
    acc += np.sum(y == tt)
    
acc = acc / x_test.shape[0]
print("test accuracy:" + str(acc))

classified_ids = np.array(classified_ids)
classified_ids = classified_ids.flatten()
 
max_view = 20
current_view = 1

fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2)

mis_pairs = {}
for i, val in enumerate(classified_ids == t_test):
    if not val:
        ax = fig.add_subplot(4, 5, current_view, xticks=[], yticks=[])
        ax.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
        mis_pairs[current_view] = (t_test[i], classified_ids[i])
            
        current_view += 1
        if current_view > max_view:
            break

print("======= misclassified result =======")
print("{view index: (label, inference), ...}")
print(mis_pairs)

plt.show()


# 인식하지 못한 이미지를 보면 사람들도 쉽게 파악하기 힘든 애매한 케이스들
# 딥러닝 학습이 사람들의 인지능력에 거의 유사하게 이미지를 인식할 수 있다는 것을 파악 가능