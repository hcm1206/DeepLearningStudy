# 기본 퍼셉트론을 이용하여 NAND 게이트 설계
# AND 게이트의 구조에서 가중치와 편향만 바꾸고 나머지는 같은 구조
# 사실 AND 게이트의 가중치와 편향을 음의 값으로 바꾸기만 하면 됨

import numpy as np

def NAND(x1, x2):
    # x에 입력 값들을 넘파이 배열로 저장
    x = np.array([x1,x2])
    # w에 임의의 가중치 값들(-0.5,-0.5)을 넘파이 배열로 저장
    w = np.array([-0.5,-0.5])
    # b에 편향 값(0.7) 저장
    b = 0.7
    # 입력 값과 가중치를 곱한 값들의 합과 편향 값을 더한 값을 tmp에 저장 (x1*w1 + x2*w2 + b)
    tmp = np.sum(x*w)+b
    # tmp가 0 이하라면 0 출력, 0 초과라면 1 출력
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

# 모든 경우의 수를 NAND 게이트에 입력하여 출력
print(NAND(0,0))
print(NAND(0,1))
print(NAND(1,0))
print(NAND(1,1))