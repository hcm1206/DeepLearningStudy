# 기계학습의 경사법에서는 기울기(경사) 값을 이용하여 매개변수를 어떤 방향으로 조절할지 결정
# 기울기, 즉 미분값은 어떤 수치의 특정 순간의 변화량을 의미

# 단순하게 미분의 정의 그대로 구현한 미분 함수

# 어떤 함수(f)의 특정 위치(x)의 값을 미분
def numerical_diff(f, x):
    h = 10e-50
    return (f(x+h)-f(x)) / h

# h 값은 가능한 한 0에 최대한 가깝게 하고자 10e-50이라는 작은 값 이용
# 하지만 이 정도로 작은 값을 컴퓨터에서 계산하면 반올림 오차로 인하여 0.0으로 취급하여 계산에 문제 발생
# h 값으로 반올림 오차가 일어나지 않도록 10**-4(1e-4) 정도의 값 사용

# 이러한 수치 미분에는 오차가 포함(h가 무한히 0으로 가까워질 수 없기 때문에 h값으로 인한 오차)
# 따라서 오차를 줄이기 위해 (x+h)와 (x-h)일 때의 함수의 차분을 계산하여 중심 차분(중앙 차분)을 구하여 미분에 사용

# 위의 2가지 개선점을 적용한 미분 함수

def numerical_diff(f, x):
    h = 1e-4
    # 왜인지는 모르겠는데 분모 부분 괄호 빼면 결과 달라짐
    return (f(x+h)-f(x-h))/(2*h)



# 간단한 함수 미분 예시
# y = 0.01x^2 + 0.1x
def function_1(x):
    return 0.01*x**2 + 0.1*x

# 함수의 입력값(x)과 출력값(f(x))을 그래프로 시각화

import numpy as np
import matplotlib.pylab as plt

# 입력값은 (0.0부터 20.0까지 0.1 간격으로 띄워진 실수 넘파이 배열)
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.show()

# x = 5일 때 함수의 미분값
print(numerical_diff(function_1, 5))
# x = 10일 때 함수의 미분값
print(numerical_diff(function_1, 10))

print()

# 계산한 미분값이 x에 대한 f(x)의 변화량, 즉 기울기
# f(x) = 0.01x^2 + 0.1x의 해석적 해는 0.02x + 0.1
# x = 5와 x = 10일 때의 미분 값은 각각 0.2, 0.3
# 수치 미분(함수로 계산한 미분)과 비교하면 매우 적은 오차


# 수치 미분값을 기울기를 구하는 함수
def tangent_line(f, x):
    # x일 때의 f 함수의 미분값(기울기)를 d에 저장
    d = numerical_diff(f, x)
    print(d)
    # 미분 값을 기울기로 하는 직선 함수 구현
    y = f(x) - d*x
    return lambda t: d*t + y

x = np.arange(0.0,20.0,0.1)
# y의 값은 함수의 출력값
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
# y2의 값은 x = 5일때 y의 미분값을 기울기로 하는 직선 함수의 출력값
y2 = tf(x)

plt.plot(x,y)
plt.plot(x,y2)
plt.show()