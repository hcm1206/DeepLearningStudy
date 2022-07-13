# 출력층에 사용되는 소프트맥스 함수 + 교차 엔트로피 함수를 하나의 계층으로 구현
# 소프트맥스 함수는 정규화함수이므로 학습할 때는 확률을 구하기 위해 사용하지만 추론할 때는 하나의 답만 구하면 되므로 소프트맥스(정규화) 함수가 필요 없음

# 소프트맥스 함수의 손실 함수로 교차 엔트로피 오차를 사용하면 역전파가 (정규화된 결과 값 yn) - (해당 인덱스 실제 정답 값 tn)으로 계산됨

import sys
# 소트프맥스 함수, 교차 엔트로피 함수 불러오기
sys.path.append("/Users/hyeon/Documents/Python/DeepLearningStudy")
from common.functions import *

# 소프트맥스-손실함수 계층 구현
class SoftmaxWithLoss:
    def __init__(self):
        # 손실 값 배열 저장할 변수 생성
        self.loss = None
        # 추론 값 배열 저장할 변수 생성
        self.y = None
        # 정답 레이블 배열 저장할 변수 생성
        self.t = None

    # 순전파 계산 (입력 값과 정답 값 배열을 입력받음)
    def forward(self,x,t):
        # 정답 값 배열을 변수로 저장(역전파 계산 때 사용하기 위해)
        self.t = t
        # 소프트맥스 함수에 입력값 배열(x)를 입력하여 나온 추론 값 배열을 self.y 변수에 저장
        self.y = softmax(x)
        # 교차 엔트로피 함수에 추론 값 배열(self.y)와 정답 레이블 배열(self.t)를 입력하여 정규화된 추론 결과(손실 값)를 변수에 저장
        self.loss = cross_entropy_error(self.y, self.t)
        # 추론 결과(손실 값) 리턴
        return self.loss

    # 역전파 계산 (결과값 미분(기본 1) 입력받음)
    def backward(self, dout=1):
        # 배치 크기 지정 (정답 값 배열의 행(데이터)의 수)
        batch_size = self.t.shape[0]
        # 입력값의 역전파(미분) 계산은 추론 값에서 정답 값을 뺀 값(배열)을 배치 크기로 나누어 진행
        dx = (self.y-self.t)/batch_size
        # 입력값의 미분값 리턴
        return dx