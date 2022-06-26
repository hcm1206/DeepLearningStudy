# 기본적인 퍼셉트론 개념을 이용하여 OR 게이트 구현
# 마찬가지로 구조는 and 게이트 및 nand 게이트와 같고 가중치와 편향값만 변경

import numpy as np

def OR(x1, x2):
    # x에 입력 값들을 넘파이 배열로 저장
    x = np.array([x1,x2])
    # w에 임의로 설정한 가중치 값들(0.5,0.5)을 넘파이 배열로 저장
    w = np.array([0.5,0.5])
    # b에 임의로 설정한 편향 값(-0.2)을 저장
    b = -0.2
    # tmp에 입력 값과 가중치를 곱한 값의 합과 편향 값을 더한 값 저장 (w1*x1 + w2*x2 + b)
    tmp = np.sum(w*x)+b
    # tmp가 0 이하이면 0 출력, 0 초과이면 1 출력
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

# 모든 경우의 수를 OR 게이트에 입력하여 출력
print(OR(0,0))
print(OR(0,1))
print(OR(1,0))
print(OR(1,1))