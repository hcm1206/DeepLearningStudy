# 신경망을 구성하는 층(노드에서 노드 사이 계층)을 각각 클래스 하나로 구현

# ReLU 함수 수식(순전파)
# y = x(x>0)
# y = 0(x<=0)

# ReLU 함수에서 x에 대한 y의 미분
# dx = 1(x>0)
# dx = 0(x<=0)

# x가 0보다 크면 상류의 값을 그대로 하류로 보내고 x가 0 이하이면 값을 아예 보내지 않음(0)

# 활성화 함수 ReLU 계층 구현
# 순전파와 역전파 계산의 입력값은 넘파이 배열을 받는다고 가정

class ReLU:
    def __init__(self):
        # 입력 배열에서 ReLU 함수의 조건을 만족하는 원소를 추려낼 배열로 사용할 변수 생성
        self.mask = None
    
    # 입력값으로 x 배열을 입력받는 순전파 구현
    def forward(self, x):
        # x 배열에서 0보다 큰 원소 위치에 True, 0 이하인 원소 위치에 False를 저장한 배열을 self.mask에 저장
        self.mask = (x <= 0)
        # out에 x와 동일한 원소를 가진 복사본 배열 저장
        out = x.copy()
        # self.mask 배열에서 같은 위치에 False인 out 배열의 원소에 0 저장(0 이하의 원소를 모두 0으로 변환)
        out[self.mask] = 0
        
        # ReLU 함수 순전파 계산 결과(out) 출력
        return out

    # 역전파 구현 
    def backward(self, dout):
        # 상류로부터 입력받은 미분값 배열 중에서 순전파 계산 결과가 0인 위치의 원소를 0으로 바꾸고 나머지 원소들은 그대로 둠
        dout[self.mask] = 0
        # 계산된 미분값을 dx에 저장
        dx = dout
        # dx 출력 (순전파에서 계산 결과가 0이었던 원소는 0으로 바꾸고 나머지는 그대로 하류로 흘려보냄)
        return dx



# ReLU 함수 순전파 계산 예시

import numpy as np

# x에 임의의 넘파이 배열 저장
x = np.array([[1.0,-0.5],[-2.0,3.0]])
# x 배열 출력
print(x)
# relu에 ReLU 계층 클래스 저장
relu = ReLU()
# ReLU 계층 순전파에 x 배열을 입력하여 출력된 값 출력(0보다 큰 원소는 그대로, 0 이하의 원소는 모두 0으로 바뀌어 출력됨)
print(relu.forward(x))
