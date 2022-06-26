# XOR 게이트는 기본적인 퍼셉트론 하나만으로 구현 불가
# 퍼셉트론 하나로 구현(구분)하려면 '선형' 영역이어야 하고 '비선형' 영역을 퍼셉트론 하나로는 구현(구분)할 수 없음
# 하지만 퍼셉트론을 층층이 쌓은 '다층 퍼셉트론'을 이용하면 XOR 게이트와 같은 '비선형' 영역을 제어 가능

# XOR 게이트는 NAND, OR, AND 게이트를 하나씩 사용하여 구현할 수 있음
# AND((NAND(x1,x2)),(OR(x1,x2)))

import numpy as np

# NAND 게이트
def NAND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

# OR 게이트
def OR(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

# AND 게이트
def AND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

# XOR 게이트 : NAND 연산 결과 값과 OR 연산 결과 값을 AND 연산한 값과 같음
def XOR(x1, x2):
    # s1에 NAND 연산 결과 값 저장
    s1 = NAND(x1,x2)
    # s2에 OR 연산 결과 값 저장
    s2 = OR(x1,x2)
    # y에 s1(NAND 연산 결과)과 s2(OR 연산 결과)를 AND 연산한 결과 저장
    y = AND(s1,s2)
    # 최종 결과(y) 출력
    return y

# 모든 경우의 수를 XOR 게이트에 입력하여 출력
print(XOR(0,0))
print(XOR(0,1))
print(XOR(1,0))
print(XOR(1,1))