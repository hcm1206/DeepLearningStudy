# 모멘텀(Momentum)은 물리 분야에서 '운동량'을 의미하는 용어
# 물리에서의 속도(velocity) 개념을 도입하여 이전 값의 속도가 매개변수 갱신에 영향을 주게끔 매개변수의 갱신값을 조정

import numpy as np

# 모멘텀 기법 클래스 정의
class Momentum:
    # 클래스 생성자에서 학습률(기본 0.01)과 모멘텀(기본 0.9)을 입력받음
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        # 속도(velocity) 변수 v를 선언하여 None으로 저장
        self.v = None

    # 매개변수 갱신 메소드(매개변수와 매개변수의 기울기를 입력받아 매개변수 갱신)
    def update(self, params, grads):
        # 매개변수가 처음으로 갱신되었을 때 실행
        if self.v is None:
            # 클래스 변수 v에 빈 딕셔너리 저장
            self.v = {}
            # params(매개변수 딕셔너리)에서 매개변수 명(key)과 실제 매개변수 값(val)을 불러와 반복
            for key, val in params.items():
                # 해당 매개변수의 속도를 일단 0으로 지정
                self.v[key] = np.zeros_like(val)

        # 각 매개변수 종류별로 반복
        for key in params.keys():
            # 해당 매개변수의 속도 계산 (설정한 모멘텀 값과 해당 매개변수의 원래 속도를 곱한 값에서 학습률과 해당 매개변수의 기울기를 곱한 값을 감산)
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            # 해당 매개변수의 값에서 계산한 해당 매개변수의 속도 값을 뺀 값을 해당 매개변수의 값으로 저장
            params[key] += self.v[key]