# 활성화 함수 : 입력받은 특정 값이 어떤 임계값을 넘는지를 판단하여 출력신호로 바꾸는 함수

# 계단 함수는 퍼셉트론에서 사용한 활성화 함수의 한 종류
# 어떤 계산한 값이 0을 넘으면 1, 아니면 0 출력

import numpy as np

# 실수 하나 만을 입력받는 계단 함수 1
def step_function1(x):
    if x > 0:
        return 1
    else:
        return 0

# 계단 함수 1의 테스트 결과 출력
print(step_function1(-0.3))
print(step_function1(0.5))
print()

# 어떤 계산한 값들을 넘파이 배열로 입력받는 계단 함수 2
def step_function2(x):
    # x의 넘파이 배열 원소들이 0보다 큰지 여부(Bool)를 y에 넘파이 배열로 저장
    y = x > 0
    # y 넘파이 배열의 True 값을 1로, False 값을 0으로 형변환(Bool -> int) 
    return y.astype(np.int)


