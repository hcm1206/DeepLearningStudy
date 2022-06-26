import numpy as np

# x는 입력값을 저장한 넘파이 배열
x = np.array([0,1])
# w는 가중치를 저장한 넘파이 배열
w = np.array([0.5,0.5])
# b는 편향(bias)
b = -0.7

# 입력값과 가중치를 각각 곱하여 넘파이 배열로 출력
print(w*x)
# 입력값과 가중치를 각각 곱한 넘파이 배열의 원소의 합 출력
print(np.sum(w*x))
# 입력값과 가중치를 각각 곱한 넘파이 배열의 원소의 합에 편향값을 더하여 출력
print(np.sum(w*x) + b)