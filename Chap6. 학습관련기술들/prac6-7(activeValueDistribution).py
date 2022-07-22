# 초기 가중치를 0으로 설정하면 다음 뉴런에 똑같은 값이 전달되기 때문에 계층을 여러개로 나누는 의미가 없음

# 가중치의 초기값은 균일한 값으로 설정하는 것이 아닌 무작위 범위의 값들로 설정해야 함
# 초기 가중치의 값이 균일하면 두 번째 가중치의 값이 모두 똑같이 갱신되기 때문에 갱신해도 똑같은 값을 유지하게 됨


# 가중치의 초기값에 따라 활성화 함수를 거친 활성화 값이 어떻게 변하는지 히스토그램으로 실험

# 1. 기본형 ==================================================================================================

import numpy as np
import matplotlib.pylab as plt

# 활성화 함수로 사용할 시그모이드 함수
def sigmoid(x):
    return 1/(1+np.exp(-x))

# 한 개에 100개의 0~1의 표준정규분포값이 저장된 데이터 1000개를 행렬로 생성하여 x(입력값)에 저장
x = np.random.randn(1000,100)
# 노드 개수를 100으로 지정
node_num = 100
# 은닉층 개수를 5로 지정
hidden_layer_size = 5
# 활성화 값을 저장할 딕셔너리 생성
activations = {}

# 은닉층 개수만큼 반복
for i in range(hidden_layer_size):
    # 현재 첫번째 은닉층이 아니라면
    if i != 0:
        # 입력값을 이전 은닉층의 활성화 값으로 설정
        x = activations[i-1]

    # 표준편차가 1이고 노드 개수(100) × 노드 개수(100) 만큼의 정규분포값 행렬을 w(가중치)로 저장
    w = np.random.randn(node_num, node_num) * 1
    # 입력값과 가중치를 행렬곱한 계산값 행렬을 a에 저장
    a = np.dot(x,w)
    # a를 시그모이드 함수에 입력한 활성화 값 행렬을 z에 저장
    z = sigmoid(a)
    # 현재 은닉층(i)의 활성화값 행렬(z)를 activations 딕셔너리에 저장
    activations[i] = z

# activations(활성화값) 딕셔너리에서 은닉층 번호(i), 활성화 값(a)을 불러와 반복
for i, a in activations.items():
    # 행을 1, 열을 activations 딕셔너리의 크기(5)로 하는 그래프 목록에서 i+1번째 위치의 그래프를 현재 그래프로 지정
    plt.subplot(1, len(activations), i+1)
    # 현재 그래프의 제목 지정
    plt.title(str(i+1) + "-layer")
    # 첫번째 그래프가 아니라면 y축 눈금과 레이블 제거
    if i != 0: plt.yticks([], [])
    # 1차원으로 바꾼 활성화 값을 입력값으로 하는 칸(열)의 수 30, 범위를 0~1로 지정한 히스토그램 생성
    plt.hist(a.flatten(), 30, range=(0,1))

plt.show()

# 히스토그램에 따르면 각 층의 활성화 값들이 0과 1에 치우쳐서 분포
# 시그모이드 함수의 출력값(활성값)이 0 또는 1에 가까워지면 그 미분값이 0에 가까워지기 때문에 역전파의 기울기 값이 점점 작아지다가 사라짐
# 이러한 현상을 기울기 소실(gradient vanishing)이라 함

# 2. 표준편차 수정 ==================================================================================================
# 가중치의 표준 편차를 0.01로 바꾸어 테스트

x = np.random.randn(1000,100)
node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
        # 표준편차만 0.01로 변경
    w = np.random.randn(node_num, node_num) * 0.01
    a = np.dot(x,w)
    z = sigmoid(a)
    activations[i] = z


for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()

# 이 경우 활성화값들이 0.5 부근에 집중 : 기울기 소실이 일어나지는 않음
# 하지만 많은 뉴런이 거의 같은 값을 출력하고 있으므로 은닉층을 여러 개로 설정한 의미가 없음
# 이러한 문제를 '표현력 제한'이라고 함
# 각 층의 활성화 값이 고르게 분포되어있어야 학습이 효율적으로 이루어지고 기울기 소실이나 표현력 제한이 일어나지 않음



# 3. Xavier 초기값 설정 =========================================================================================
# 가중치 초깃값을 Xavier 초깃값으로 설정하여 테스트
# Xavier 초기값은 앞 계층의 노드가 n개일 시 표준편차가 1/n^^(1/2)인 분포를 사용하는 방식

x = np.random.randn(1000,100)
node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
    # 표준편차를 노드 개수의 제곱근으로 변경
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
    a = np.dot(x,w)
    z = sigmoid(a)
    activations[i] = z


for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    plt.ylim(0, 6000)
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()

# 첫번째 층에서는 활성화값이 정규분포에 가깝게 분포하고, 층이 깊어질수록 정규분포에서 다소 일그러지는 형태로 분포
# 그러나 상기 방식(#1, #2)보다는 넓게 분포하여 학습이 상대적으로 효율적으로 이루어 질 것
# 일그러짐을 최소화하려면 시그모이드 함수 대신 쌍곡선 함수인 tanh 함수를 사용하여 개선 가능


# 4. ReLU 함수를 활성화 함수로 사용할 때 =========================================================================================
# 활성화 함수로 ReLU 함수를 사용할 때에는 ReLU에 특화된 초기값인 He 초깃값 사용 권장
# He 초깃값은 앞 계층의 노드가 n개일 때 표준편차가 2/n**(1/2)인 정규분포 사용
# ReLU는 음의 영역이 0이므로 더 넓게 분포시키고자 대략 2배의 계수가 필요하기 때문

# ReLU 함수 정의
def ReLU(x):
    return np.maximum(0, x)


# 4-1. ReLU 함수 + 표준 편차 0.01인 정규분포 가중치

x = np.random.randn(1000,100)
node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
    # 가중치 초기값은 표준편차 0.01인 임의의 정규분포값
    w = np.random.randn(node_num, node_num) * 0.01
    a = np.dot(x,w)
    # 활성화 함수는 ReLU 함수
    z = ReLU(a)
    activations[i] = z


for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()

# 활성화 값들이 아주 작은 값들이므로 역전파시의 기울기가 작아져 학습이 거의 이루어지지 않음


# 4-2 ReLU 함수 + Xavier 가중치 초깃값

x = np.random.randn(1000,100)
node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
    # 가중치 초기값은 Xavier 초기값
    w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    a = np.dot(x,w)
    # 활성화 함수는 ReLU 함수
    z = ReLU(a)
    activations[i] = z


for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()

# 층이 깊어지면서 치우침이 조금씩 커지고 학습시 '기울기 소실'문제를 일으킴


# 4-3. ReLU 함수 + He 가중치 초깃값

x = np.random.randn(1000,100)
node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
    # 가중치 초기값은 He 초기값
    w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)
    a = np.dot(x,w)
    # 활성화 함수는 ReLU 함수
    z = ReLU(a)
    activations[i] = z


for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()

# 모든 층에서 활성화 값이 균일하게 분포되어 역전파 때 기울기 값도 적절하게 도출될 것으로 예상