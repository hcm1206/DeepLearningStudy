# 신경망에서 손실함수가 최솟값이 되도록 하는 최적의 가중치, 편향을 찾아내야 함
# 기울기를 활용하여 손실함수가 최솟값이 되도록 하는 지점을 잘 찾아야 함
# 하지만 기울기가 가리키는 곳은 항상 최솟값이라는 보장이 없고 안장점 또는 극솟값일 수도 있음(최솟값, 극솟값, 안장점에서의 기울기는 0)


# 현재 지점에서 기울기가 가리키는(기울어진) 방향으로 일정지점만큼 이동하여 이동한 지점의 기울기를 구하는 과정을 반복하여 함수의 값을 줄여나가는 방법을 경사법(gradient method)이라 함
# 함수의 최솟값을 찾을 때를 경사 하강법, 함수의 최댓값을 찾을 떄를 경사 상승법이라고 하며 신경망에서는 주로 경사하강법 사용

# 학습률 : 한 번의 학습에 진행되는 매개변수 갱신의 횟수

import numpy as np

# 편미분 함수
def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad


# 경사 하강법 구현

# f는 최솟값을 찾기 위해 최적화할 함수, init_x는 최초로 입력해볼 x값, lr은 학습률(기본값 0.01), step_num은 경사법에 따른 반복 횟수(기본값 100)
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    # 먼저 x에 최초 입력값 init_x 저장
    x = init_x

    # step_num(입력한 반복 횟수)만큼 반복
    for i in range(step_num):
        # f 함수에서 x 값일때의 미분값(기울기)를 grad에 저장
        grad = numerical_gradient(f,x)
        # x에 lr과 grad를 곱한 값을 빼서 저장 (grad에서 나온 기울기에 학습률(lr)을 가중치로 곱하여 x값에서 뺀 값을 새로운 x 입력값으로 지정)
        x -= lr*grad
    # 최종적으로 나온 x값 리턴
    return x

# 입력값이 2개인 임의의 함수
def function_2(x):
    return x[0]**2 + x[1]**2

# 최초로 입력할 x 초기값 (-3.0, 4.0)
init_x = np.array([-3.0,4.0])
# 최초 x입력값을 init_x(-3.0,4.0), 학습률을 0.1, 경사법 반복 횟수를 100으로 설정한 경사법을 통해 function_2 함수의 최솟값을 만드는 x 값을 구하여 출력
print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))

# 결과는 진정한 최솟값인 (0,0)과 거의 같은 값이 나옴

# 위 과정을 그림으로 나타내기

import matplotlib.pylab as plt

# 그림으로 명확하게 나타내기 위해 기울기 함수 코드 일부 변형
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    # 이전 x 값을 저장하기 위한 x_history 리스트를 생성
    x_history = []

    for i in range(step_num):
        # x 값을 복사하여 x_history 리스트에 원소로 저장
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad

    # 결과값 x와 이전 x 배열들이 저장된 x_history 리스트를 튜플로 리턴
    return x, np.array(x_history)


lr = 0.1
step_num = 20
# init_x 초기화
init_x = np.array([-3.0,4.0])
# x에 경사법 결과, x_history에 경사법 과정에서 사용된 x 값들의 배열(리스트) 저장
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)
# x축 -5~5, y축 0~0에 파란색 점선 그리기
plt.plot( [-5, 5], [0,0], '--b')
# x축 0~0, y축 -5~5에 파란색 점선 그리기
plt.plot( [0,0], [-5, 5], '--b')
# x_history의 각각의 좌표 위치에 원 그리기
plt.plot(x_history[:,0], x_history[:,1], 'o')

# x축 범위 : -3.5 ~ 3.5
plt.xlim(-3.5, 3.5)
# y축 범위 : -4.5 ~ 4.5
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()




# 학습률이 너무 크거나 너무 작으면 좋지 않은 결과 도출

# 기울기 함수 원상 복구
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr*grad
    return x

# 학습률이 너무 클 때(lr = 10.0)
init_x = np.array([-3.0,4.0])
print(gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100))
# 큰 값으로 발산해버림

# 학습률이 너무 작을 때(lr = 1e-10(0.0000000001))
init_x = np.array([-3.0,4.0])
print(gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100))
# 갱신이 잘 이뤄지지 않음


# 학습률과 같은 사람이 직접 설정하는 매개변수를 하이퍼파라미터(hyper parameter)라 함
# 하이퍼파라미터는 훈련데이터와 알고리즘에 의해 자동으로 획득되는 가중치, 편향 등의 신경망 매개변수와 구분됨