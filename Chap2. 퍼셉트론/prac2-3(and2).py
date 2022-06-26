# AND 게이트를 넘파이 배열과 편향 값을 이용하여 약간 변형


import numpy as np

def AND(x1, x2):
    # x에 입력값을 넘파이 배열로 저장
    x = np.array([x1,x2])
    # w에 임의의 가중치를 넘파이 배열로 저장
    w = np.array([0.5,0.5])
    # 편향값을 임의로 -0.7로 설정하여 저장
    b = -0.7
    # tmp에 입력값과 가중치를 곱한 값들의 합에 편향을 더하여 저장 (x1*w1 + x2*w2 + b)
    tmp = np.sum(x*w)+b
    # tmp가 0 이하이면 0, 0 초과이면 1 출력
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

# AND 게이트에 모든 경우의 수를 입력하여 결과 출력
print(AND(0,0))
print(AND(0,1))
print(AND(1,0))
print(AND(1,1))