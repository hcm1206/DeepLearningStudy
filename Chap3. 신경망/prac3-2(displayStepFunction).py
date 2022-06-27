# 넘파이 배열 입력받아 계단 함수에 입력한 결과 값을 그래프로 시각화

import numpy as np
import matplotlib.pylab as plt

# 넘파이 배열을 입력받는 계단 함수
def step_function(x):
    # x의 각 원소가 0보다 큰지 여부(Bool)를 정수형(0 또는 1)으로 전환하여 넘파이 배열로 저장 후 리턴
    return np.array(x > 0, dtype=int) # 책에 dtype=np.int로 적혀있는데 경고창에 의하면 걍 int로 쓰라고 권장

# x에 -5.0부터 5.0 직전 값까지 0.1 간격의 원소들을 저장한 넘파이 배열 생성하여 저장
x = np.arange(-5.0,5.0,0.1)
# 넘파이 배열 x를 계단 함수(활성화 함수)에 입력하여 나온 (0 또는 1의 활성화 여부가 저장된) 넘파이 배열을 y에 저장
y = step_function(x)
# x,y를 각각 x축 y축으로 하여 그래프로 시각화
plt.plot(x,y)
# y축 범위를 -0.1부터 1.1까지 설정
plt.ylim(-0.1,1.1)
plt.show()
