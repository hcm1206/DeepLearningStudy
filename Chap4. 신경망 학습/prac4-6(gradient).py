# 변수가 2개 이상인 함수에서 모든 변수의 편미분을 벡터로 정리한 것을 기울기(gradient)라고 함

# 편미분 함수 구현

import numpy as np


# 편미분 함수 (f는 편미분을 진행할 함수, x는 편미분을 할 x값(위치)이 저장된 넘파이 배열)
def numerical_gradient(f,x):
    # 미분을 위한 0에 가까운 극한 값(0.0001으로 지정)을 h에 저장
    h = 1e-4
    # x 넘파이 배열과 크기(원소 수)가 같은 0으로 이루어진 넘파이 배열을 grad에 저장
    grad = np.zeros_like(x)

    # x 배열의 크기(원소 수)만큼 idx 반복(모든 x값에 대하여 아래 연산 실행)
    for idx in range(x.size):
        # tmp_val에 원본 idx번째 x값 저장
        tmp_val = x[idx]
        
        # idx번째 x값에 원본 idx번째 x값에 극한 값을 더하여 저장
        x[idx] = tmp_val + h
        # fxh1에 idx번째 x값에 극한값이 더해진 값을 f 함수에 입력한 결과값 저장
        fxh1 = f(x)

        # idx번째 x값에 원본 idx번째 x값에 극한값을 빼서 저장
        x[idx] = tmp_val - h
        # fxh2에 idx번째 x값에 극한값이 감산된 값을 f 함수에 입력한 결과값 저장
        fxh2 = f(x)

        # grad 배열의 [idx]번째 인덱스 원소의 미분값 저장
        grad[idx] = (fxh1 - fxh2) / (2*h)
        # idx번째 x값을 원본값으로 복구
        x[idx] = tmp_val

    # 계산된 미분값 배열 grad 리턴
    return grad


# 임의의 함수 function_2 정의(f(x) = x0^2 + x1^2)
def function_2(x):
    return x[0]**2 + x[1]**2


# function_2 함수에서의 임의의 세 점 (3,4),(0,2),(3,0)에서의 기울기(편미분) 구해서 출력
print(numerical_gradient(function_2,np.array([3.0,4.0])))
print(numerical_gradient(function_2,np.array([0.0,2.0])))
print(numerical_gradient(function_2,np.array([3.0,0.0])))


# 편미분(기울기)에 -(마이너스)를 붙인 벡터를 격자 좌표 상에서 나타내기

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

# 배치가 존재하지 않는 1차원 배열의 편미분 구하기
def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val
        
    return grad

# 배치가 존재하는 다차원 배열도 계산 가능한 편미분 구하는 함수
def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        # X 배열을 반복하며 X 배열의 인덱스를 idx에, 해당 인덱스의 실제 배열 값을 x에 대입
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad

# 이 파일이 인터프리터에서 직접 실행되었을 경우에만 실행
if __name__ == '__main__':
    # x0에 -2부터 2.5 미만까지 0.25 간격의 수들을 넘파이 배열로 저장
    x0 = np.arange(-2, 2.5, 0.25)
    # x1에 -2부터 2.5 미만까지 0.25 간격의 수들을 넘파이 배열로 저장
    x1 = np.arange(-2, 2.5, 0.25)
    # x축을 x0 배열, y축을 x1 배열로 하는 격자 위치의 x축 좌표, y축 좌표를 각각 X, Y에 배열로 저장
    X, Y = np.meshgrid(x0, x1)
    # X, Y를 1차원 배열로 전환
    X = X.flatten()
    Y = Y.flatten()
    # X, Y 배열에 저장된 각각의 x좌표 y좌표 값 배열을 행과 열을 바꿔 [x,y] 형식이 되도록 수정한 값을 편미분하여 다시 행과 열을 바꾼 후 grad에 저장(배열)
    grad = numerical_gradient(function_2, np.array([X, Y]).T).T


    plt.figure()
    # X,Y(배열) 위치에서 (x,y)위치의 음수 편미분값(배열)을 향하여 방향을 반전시킨 화살표를 #666666 색상의 화살표로 그리기
    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    # 그래프에 격자 그리기
    plt.grid()
    plt.draw()
    plt.show()


# 이 그래프에서 각 화살표는 각 지점에서 낮아지는 방향을 가리킴
# 즉 기울기(화살표)가 가리키는 방향은 각 지점에서 함수 출력값을 가장 크게 줄이는 방향