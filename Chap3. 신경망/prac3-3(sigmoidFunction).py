# 활성화 함수 중에서 많이 사용하는 것이 시그모이드 함수
# 실수 1개를 입력받아 활성화 값을 출력하는 시그모이드 함수 구현
# 시그모이드 함수는 0과 1만 출력하는 것이 아닌 연속적인 값을 출력

import numpy as np
import matplotlib.pylab as plt

# 시그모이드 함수 
def sigmoid(x):
    # 넘파이 배열로 입력받으면 각 원소의 계산 결과를 넘파이 배열로 출력
    return 1 / (1 + np.exp(-x))

# 입력값으로 넣을 값들을 넘파이 배열로 x에 저장
x = np.array([-1.0,1.0,2.0])
# 시그모이드 함수에 x 넘파이 배열의 값들을 각각 입력하여 계산
print(sigmoid(x))

# 시그모이드 함수 그래프 시각화
x = np.arange(-5.0,5.0,0.1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()
