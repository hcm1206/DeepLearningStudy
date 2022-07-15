# 신경망 학습에서는 학습률 값에 따라 학습 효율이 달라짐(너무 작으면 학습 시간이 길어지고 너무 크면 학습이 제대로 이뤄지지 않음)
# 학습률 값은 처음부터 끝까지 일정할 수도 있지만 학습이 진행됨에 따라 학습률을 점차 줄여가며 학습 범위를 유동적으로 조절 가능
# 이를 학습률 감소(learning rate decay)라 하며 처음에는 크게 학습하다 학습이 진행되며 조금씩 작게 학습하는 방식
# AdaGrad 방식은 각각에 매개변수의 학습 진행에 따라 각각의 맞춤형 학습률 값을 지정해주는 방식
# 매개변수가 크게 이동했다면 학습률을 그만큼 줄여서 이동 범위를 고르게 함

# h 변수(매개변수의 기울기의 제곱)를 이용하여 기울기가 클수록 학습률을 작아지게 만듦

import numpy as np

# AdaGrad 기법 클래스 정의
class AdaGrad:
    # 초기 학습값을 입력받으며(기본 0.01) 클래스 생성
    def __init__(self,lr=0.01):
        # 학습값 클래스 변수에 입력받은 학습값 저장
        self.lr = lr
        # 초기 h값에 None 지정
        self.h = None

    # 매개변수와 기울기를 입력받아 매개변수 갱신
    def update(self,params,grads):
        # 매개변수가 처음으로 갱신되었다면
        if self.h is None:
            # h 변수에 빈 딕셔너리 저장
            self.h = {}
            # 매개변수 딕셔너리에서 매개변수 명(key)과 실제 매개변수 값(value)를 불러와 반복
            for key, val in params.items():
                # 해당 매개변수의 h 값을 일단 0으로 지정
                self.h[key] = np.zeroes_like(val)

        # 각 매개변수 반복
        for key in params.keys():
            # 해당 매개변수의 h 변수에 해당 매개변수 기울기의 제곱 저장
            self.h[key] += grads[key] * grads[key]
            # 해당 매개변수의 값에서 학습률과 기존 매개변수의 값을 곱하고 h의 제곱근 값에서 아주 작은 값(0으로 나누기를 해결하기 위함)을 더한 값을 뺀 값을 
            # 해당 매개변수의 값으로 저장
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)