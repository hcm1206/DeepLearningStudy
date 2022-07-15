# 이전 값의 속도가 매개변수에 영향을 주는 모멘텀과 매개변수의 원소마다 갱신 값을 다르게 하는 Adagrad의 개념을 합친 것이 Adam 기법
# 모멘텀과 Adagrad의 이점을 조합
# 하이퍼파리미터의 '편향 보정' 진행

# ===자세한 내용은 원논문을 참고하고 깊게 설명 안 하겠답니다 나도 자세히 안 볼거임 ㅅㄱ===

import numpy as np

# Adam 기법 클래스 정의
class Adam:

    # 학습률(기본 0.001), 베타1값(기본 0.9), 베타2값(기본 0.999)을 입력받음
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        # 입력받은 변수값들을 클래스 변수에 대입
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        # 반복 횟수를 0으로 지정
        self.iter = 0
        # m 변수 생성
        self.m = None
        # v 변수 생성
        self.v = None
    
    # 매개변수 갱신
    def update(self, params, grads):
        # 매개변수가 처음으로 갱신되었다면
        if self.m is None:
            # m 변수와 v 변수를 저장할 딕셔너리 생성
            self.m, self.v = {}, {}
            # 각 매개변수의 m 변수와 v 변수를 0으로 지정
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        # 반복 횟수 1 증가
        self.iter += 1
        # 학습률 갱신값 계산
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        # 매개변수 별로 반복
        for key in params.keys():
            # 해당 매개변수의 m 변수 계산
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            # 해당 매개변수의 v 변수 계산
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            # 해당 매개변수의 학습률 갱신값과 m 변수, v 변수를 이용하여 매개변수 갱신
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            