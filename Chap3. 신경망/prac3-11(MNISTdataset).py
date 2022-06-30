# 실전 신경망 추론 과정 구현 (손글씨 분류)
# 학습과정은 생략(이미 학습된 매개변수를 사용), 추론 과정만 구현
# 신경망은 두 단계를 거쳐 데이터 학습 (1. 훈련데이터로 가중치 매개변수 학습, 2. 학습한 매개변수를 사용하여 입력 데이터를 분류하는 추론)

# MNIST 데이터 셋(기계학습 훈련용 손글씨 숫자 이미지 집합) 사용
# 0~9의 훈련용 숫자 이미지 60,000장, 시험 이미지 10,000장

from mnist import load_mnist

# (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블) 불러오기
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# normalize : 이미지 픽셀 값을 0.0~1.0으로 정규화할 지 0~255의 값으로 유지할지 결정
# flatten : 이미지를 1차원 배열로 평탄화할지 3차원 배열로 저장할지 결정
# one_hot_label : 원-핫-레이블(정답을 뜻하는 원소만 1, 나머지는 0인 배열 형태)로 출력할 지 숫자 형태의 레이블로 출력할지 결정

# 데이터 행렬의 형상 출력
# 훈련 이미지 행렬(이미지 수, 이미지 픽셀 수)
print(x_train.shape)
# 훈련 이미지 레이블 배열(이미지 수)
print(t_train.shape)
# 시험 이미지 행렬(이미지 수, 이미지 픽셀 수)
print(x_test.shape)
# 시험 이미지 레이블 배열(이미지 수)
print(t_test.shape)


print()

# 불러온 이미지 세트 중 첫번째 이미지 출력하기

import numpy as np
from PIL import Image

# 배열에 저장된 이미지 객체를 실제 이미지로 변환하여 출력하는 과정
def img_show(img):
    # 넘파이 배열로 저장된 이미지 객체를 PIL용 이미지 데이터 객체로 변환
    pil_img = Image.fromarray(np.uint8(img))
    # 변환한 이미지 데이터 객체를 이미지화하여 출력
    pil_img.show()

# img에 0번째 인덱스 훈련용 이미지 객체 배열 저장
img = x_train[0]
# label에 0번째 인덱스 훈련용 이미지 레이블 저장
label = t_train[0]
print(label)

# 1차원으로 저장된 이미지 픽셀을 28×28 크기의 행렬로 재배열
print(img.shape)
img = img.reshape(28,28)
print(img.shape)

# img에 저장된 이미지 객체 행렬을 실제 이미지로 출력
img_show(img)


