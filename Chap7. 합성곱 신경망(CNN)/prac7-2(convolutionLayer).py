# im2col 함수를 이용하여 합성곱 계층 구현

import sys
sys.path.append("/Users/hyeon/Documents/Python/DeepLearningStudy")
from common.util import im2col, col2im
import numpy as np

# 합성곱 계층을 클래스로 정의
class Cnovolution:
    # 가중치(필터값) 행렬, 편향 행렬, 스트라이드(기본값 1), 패딩(기본값 0)을 입력받아 클래스 변수로 변환
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    # 합성곱 순전파 계산 (입력값 행렬 입력받음)
    def forward(self,x):
        # 가중치(필터값) 행렬 형상으로부터 필터값의 수, 채널 수, 필터값의 높이, 필터값의 너비 불러옴
        FN, C, FH, FW = self.W.shape
        # 입력값 행렬 형상으로부터 데이터의 수, 채널 수, 데이터의 높이, 데이터의 너비 불러옴
        N, C, H, W = x.shape
        # 데이터 높이, 필터값 높이, 패딩, 스트라이드를 이용하여 출력값 높이 계산
        out_h = int(1+(H+2*self.pad-FH)/self.stride)
        # 데이터 너비, 필터값 너비, 패딩, 스트라이드를 이용하여 출력값 너비 계산
        out_w = int(1+(W+2*self.pad-FW)/self.stride)

        # im2col 함수를 통해 입력값 행렬(x)을 2차원 행렬로 변환하여 col에 저장
        col = im2col(x, FH, FW, self.stride, self.pad)
        # 필터값 행렬에서 행의 수가 필터값의 수가 되도록 열의 수를 조절하여 2차원 행렬로 만들고 전치(행의 수와 열의 수를 뒤바꿈)하여 필터값의 2차원 행렬 저장
        col_W = self.W.reshape(FN, -1).T
        # 입력값 2차원 행렬과 필터값 2차원 행렬을 행렬곱한 후 편향값을 더하여 출력값 계산
        out = np.dot(col, col_W) + self.b
        # 계산된 2차원 출력값 행렬을 (데이터의 수, 출력값 높이, 출력값 너비, 남은 적당한 값) 4차원 행렬로 변환 후 (0번째 차원, 3번째 차원, 1번쨰 차원, 2번째 차원) 순으로 차원 재배열
        out = out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2)
        # 최종 출력값 리턴
        return out

    # 합성곱 역전파 계산 (이전 계층의 역전파 값 입력받음)
    def backward(self, dout):
        # 가중치(필터값) 행렬 형상으로부터 필터값의 수, 채널 수, 필터값의 높이, 필터값의 너비 불러옴
        FN, C, FH, FW = self.W.shape
        # 이전 계층에서의 역전파 행렬을 (0번째 차원, 2번째 차원, 3번째 차원, 1번째 차원) 순으로 차원을 배열한 후 열의 수가 필터값의 수가 되도록 행의 수를 조절하여 2차원 행렬로 만듦
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        # 편향 역전파 계산(이전 계층에서의 역전파 행렬의 모든 원소를 더함)
        self.db = np.sum(dout, axis=0)
        # 가중치(필터값) 역전파 계산(입력값의 2차원 행렬을 전치한 행렬과 이전 계층에서의 역전파 행렬을 행렬곱)
        self.dW = np.dot(self.col.T, dout)
        # 가중치 역전파 계산값을 (1차원, 0차원)순으로 재배열 후 (필터값 수, 채널 수, 필터값 높이, 필터값 너비) 형상의 4차원 행렬로 변환
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # 입력값 2차원 행렬 역전파 계산(이전 계층의 역전파 행렬값과 입력값의 2차원 행렬을 전치한 행렬을 행렬곱)
        dcol = np.dot(dout, self.col_W.T)
        # 입력값 행렬 역전파 계산 (col2im 함수 이용)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        # 입력값 행렬 역전파값 리턴
        return dx