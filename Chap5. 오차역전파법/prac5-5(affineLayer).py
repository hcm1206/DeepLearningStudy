# 신경망 순전파에서는 입력값 행렬과 가중치 행렬의 곱에 편향 행렬을 더하여 활성화 함수에 입력할 행렬값을 계산
# 이를 기하학에서 어파인 변환(affine transformation)이라고 하며 이 계산을 수행하는 처리(노드 or 계층)을 Affine 계층이라는 이름으로 구현
# 행렬곱 계산에서는 순전파 계산 역전파 계산 모두 행렬의 형상에 유의

# Affine 계층 순전파

import numpy as np

# 입력값(형상이 (2)인 행렬)
X = np.random.rand(2)
# 가중치(형상이 (2,3)인 행렬)
W = np.random.rand(2,3)
# 편향(형상이 (3)인 행렬)
B = np.random.rand(3)

print(X.shape)
print(W.shape)
print(B.shape)

# 출력값(입력값(2)×가중치(2,3)+편향(3) 행렬계산을 하여 형상이 (3)인 행렬)
Y = np.dot(X,W) + B
print(Y.shape)

print()

# 배치용 Affine 계층에서는 배치 데이터의 크기(N)에 따라 입력값 (N, x)행렬, 출력값 (N, 3)행렬로 계산
# Affine 계층 순전파 계산에서 편향 덧셈은 각 데이터에 편향값이 각각 더해지는 것

# 입력값(X)와 가중치(W)가 행렬곱 계산되었다고 가정된 임의의 2×3 형상 행렬
X_dot_W = np.array([[0,0,0],[10,10,10]])
# 3 형상의 편향 행렬
B = np.array([1,2,3])
# 입력값과 가중치가 행렬곱 계산된 행렬 출력
print(X_dot_W)
# 행렬곱 계산된 행렬에 편향값을 더한 행렬 출력(계산된 행렬의 각 원소에 대응되는 위치의 편향값 원소를 더함)
print(X_dot_W+B)

print()

# Affine 계층 역전파 계산에서는 각 데이터의 역전파 값이 편향의 원소에 모여야 함(입력받은 미분값의 1번 데이터 1번째 원소 + 2번 데이터 1번째 원소 = 미분된 편향값 1번째 원소)
# 행렬곱 역전파 계산은 행렬을 전치행렬로 바꾼 후(ex 2×3 => 3×2) 계산해야 함

# 임의의 미분된 출력값으로 가정하는 2×3 형상 행렬
dY = np.array([[1,2,3],[4,5,6]])
print(dY)
# 편향 미분값 계산 (dY 행렬에서 각 데이터의 같은 위치의 원소를 합한 행렬)
dB = np.sum(dY, axis=0)
print(dB)


# Affine 계층 구현
class Affine:
    # 계산 과정에 사용할 가중치와 편향을 입력받음
    def __init__(self,W,b):
        # 가중치를 저장할 인스턴스 변수 생성 후 입력값 저장
        self.W = W
        # 편향을 저장할 인스턴스 변수 생성 후 입력값 저장
        self.b = b
        # 순전파 계산시 입력되는 입력값을 저장할 인스턴스 변수 미리 생성
        self.x = None
        # 역전파 계산시 사용할 가중치 미분값을 저장할 인스턴스 변수 미리 생성
        self.dW = None
        # 역전파 계산시 사용할 편향 미분값을 저장할 인스턴스 변수 미리 생성
        self.db = None
    
    # 순방향 계산 (x는 입력값 배열)
    def forward(self,x):
        # 입력값 배열을 인스턴스 변수로 저장
        self.x = x
        # 입력값 배열과 가중치 배열을 행렬곱한 값에 편향 행렬을 더한 값 out 저장
        out = np.dot(x,self.W) + self.b
        # out 출력
        return out

    # 역전파 계산 (dout은 출력값 미분 배열)
    def backward(self,dout):
        # 입력값 미분은 상류에서 받아온 미분값 행렬과 가중치 행렬의 전치행렬을 행렬곱하여 구함
        dx = np.dot(dout, self.W.T)
        # 가중치 미분은 입력값 행렬의 전치행렬을 상류에서 받아온 미분값 행렬과 행렬곱하여 구함
        self.dW = np.dot(self.x.T, dout)
        # 편향 미분은 상류에서 받아온 미분값 행렬에서 각 배열의 같은 인덱스의 값을 모두 합하여 구한 배열
        self.db = np.sum(dout, axis=0)
        # 입력값 미분 리턴
        return dx