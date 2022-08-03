# 풀링 계층은 입력 데이터 행렬에서 특정 범위의 값 중 가장 큰 값을 모아 행렬로 만드는 과정
# 먼저 입력 데이터를 전개하여 지정하고싶은 범위를 하나의 행으로 만들고, 그 행에서 가장 큰 값을 모아 행렬로 만든 후 그 행렬을 변형하여 다차원 행렬로 만들면 됨

import sys
sys.path.append("/Users/hyeon/Documents/Python/DeepLearningStudy")
from common.util import im2col, col2im
import numpy as np

# 풀링 계층 클래스 구현
class Pooling:
    # 풀링값 높이, 풀링값 너비, 스트라이드(기본 1), 패딩(기본 0) 입력받아 클래스 변수로 저장
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        # 입력값 행렬 저장할 공간 생성
        self.x = None
        # 입력값 행렬의 지정 범위에서의 최대값들을 저장할 행렬 공간 생성
        self.arg_max = None

    # 풀링 순전파 계산(입력값 행렬 입력받음)
    def forward(self, x):
        # 입력값 행렬에서 데이터의 수, 채널 수, 데이터 높이, 데이터 너비 불러옴
        N, C, H, W = x.shape
        # 출력값 높이 계산
        out_h = int(1 + (H - self.pool_h) / self.stride)
        # 출력값 너비 계산
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # im2col 함수를 이용하여 입력값의 2차원 행렬 계산 후 col에 저장
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # col 행렬의 열이 풀링값 높이와 풀링값 너비의 곱이 되도록 행렬을 재배열
        col = col.reshape(-1, self.pool_h*self.pool_w)

        # col 행렬의 각 1차원의 값들 중 가장 큰 값을 뽑아 행렬로 저장
        arg_max = np.argmax(col, axis=1)
        # col 행렬의 1차원에서의 최대값을 출력값으로 지정
        out = np.max(col, axis=1)
        # 출력값을 (데이터 수, 출력값 높이, 출력값 너비, 채널 수) 형상의 4차원 행렬로 변환 후 (0,3,1,2) 순으로 차원 순서를 바꿈
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        # 입력값과 최대값 행렬을 클래스 변수로 저장
        self.x = x
        self.arg_max = arg_max

        # 최종 출력값 리턴
        return out

    # 풀링 계층 순전파 계산은 크게 3단계로 구성
    # 1. 입력 데이터 전개(다차원 데이터를 2차원으로 전개)
    # 2. 행별 최댓값을 추출(지정한 풀링 영역마다 하나의 행으로 들어가 있기 때문)
    # 3. 추출한 최댓값 행렬을 적절하게 성형(1차원 행렬을 다시 데이터(배치) 수, 채널 수에 맞게 다차원 데이터로 변형)
    
    # 풀링 계층 역전파 계산(이전 계층에서 받아온 역전파 값 입력받음)
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        # 풀링 영역 크기를 풀링 높이, 풀링 너비의 곱으로 구함
        pool_size = self.pool_h * self.pool_w
        # 최대값 역전파값을 저장할 (입력받은 역전파값의 크기, 풀링 영역 크기) 형상의 행렬 생성
        dmax = np.zeros((dout.size, pool_size))
        # (0~최대값 행렬의 크기의 행렬, 최대값 행렬을 1차원으로 변환한 행렬) 형상의 최대값 역전파값 행렬에 입력된 역전파값의 1차원 행렬을 저장
        # 솔직히 이해 불가
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        # 최대값 역전파값을 최대값 역전파값의 형상과 풀링 영역 크기를 곱한 값으로 행렬 재배열
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        # 출력 행렬의 역전파 값을 최댓값 역전파값의 0차원, 1차원, 2차원 값의 곱을 행으로 하여 남은 값을 열로 하는 행렬값으로 계산
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        # col2im 함수를 이용하여 입력값 배열의 역전파 값 계산
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        # 입력값 배열의 역전파 값 리턴
        return dx
