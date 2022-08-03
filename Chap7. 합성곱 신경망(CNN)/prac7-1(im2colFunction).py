# 합성곱 신경망(CNN)에서 계층 간에 전달되는 데이터는 4차원 데이터
# 어떤 4차원 데이터의 형상이 (10,1,28,28)이라면 높이 28, 너비 28, 채널(폭) 1인 데이터가 4개 존재한다는 뜻

import numpy as np

# 형상이 10,1,28,28인 4차원 데이터 예시 
x = np.random.rand(10,1,28,28)
# x는 각 원소에 0~1 사이의 랜덤 실수가 저장된 4차원 행렬 데이터
print(x)
# x 데이터의 형상
print(x.shape)

# 첫번째 x 데이터의 형상
print(x[0].shape)
# 두번째 x 데이터의 형상
print(x[1].shape)

# 첫번째 x 데이터와 두번째 x 데이터의 형상은 같음(높이 28, 너비 28, 채널(폭) 1인 데이터가 모인 것이 x이고, x에서 몇번째 데이터이건 형상은 같음)

# 첫번째 데이터의 첫번째 채널(폭)의 공간 데이터(2차원 행렬) 출력
print(x[0,0])
print()


# 합성곱 연산을 클래스로 구현
# for문 대신 트릭을 사용한 im2col 함수 사용 (image to column : 이미지->행렬 변환이라는 뜻)

# 3차원 데이터(배치를 적용했다면 4차원)를 필터링(가중치 계산)하기 좋게끔 2차원 평면 데이터로 바꾸는 함수가 im2col 함수
# 3차원 데이터 상에서 필터를 적용하고 싶은 특정 데이터 영역을 펼쳐서 하나의 행으로 만들고, 다른 영역들도 마찬가지로 또다른 행으로 펼치며 2차원으로 만드는 것

# im2col 함수 정의 예시

# 입력값 행렬, 필터값 행렬 높이, 필터값 행렬 너비, 스트라이드(기본값 1), 패딩(기본값 0) 입력받는 im2col 함수 정의
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    # 입력값 행렬의 형상으로부터 각 요소(데이터의 수(N), 채널(C), 높이(H), 너비(W))를 받아옴
    N, C, H, W = input_data.shape
    # 출력값 행렬의 높이 계산 : (입력값 높이 + 패딩의 2배 값 - 필터값 높이)를 스트라이드로 나눈 몫에 1을 더한 값
    out_h = (H + 2*pad - filter_h)//stride + 1
    # 출력값 행렬의 너비 계산 : (입력값 너비 + 패팅의 2배 값 - 필터값 너비)를 스트라이드로 나눈 몫에 1을 더한 값
    out_w = (W + 2*pad - filter_w)//stride + 1
    # 입력값 이미지에 패딩을 넣어 저장 (4차원 입력값 데이터에서 데이터의 수(N), 채널 수(C)에는 패딩을 추가하지 않고 입력값 높이(H), 입력값 너비(W)에 입력받은 패딩값(pad)을 패딩으로 추가)
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    # 결과값 행렬을 저장하기 위하여 (데이터의 수, 채널 수, 필터값 높이, 필터값 너비, 출력값 높이, 출력값 너비) 형상의 행렬 생성
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # 필터값 높이만큼 반복
    for y in range(filter_h):
        # y값의 최대값은 y에 (스트라이드 값과 출력값 높이)를 더한 값
        y_max = y + stride*out_h
        # 필터값 너비만큼 반복
        for x in range(filter_w):
            # x값의 최대값은 x에 (스트라이드 값과 출력값 너비)를 더한 값
            x_max = x + stride*out_w
            #
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    # col 값의 0,1,2,3,4,5번째 차원의 형상을 0,4,5,1,2,3 순으로 바꾸고 2차원 행렬로 형상 변경
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    # 최종 col값 리턴
    return col

# 이해 불가

# im2col 함수를 이용하여 실제로 4차원 데이터 행렬을 2차원 행렬로 바꿔보기

# 0차원(N, 데이터 수)에 1개, 1차원(C, 채널)에 3개, 2차원(H, 데이터 높이)에 7개, 3차원(W, 데이터 너비)에 7개의 0~1 사이 랜덤 실수 원소를 갖는 입력값 행렬 x1
x1 = np.random.rand(1, 3, 7, 7)
# im2col 함수를 이용하여 x1 입력 데이터를 높이 5, 너비 5의 필터, 스트라이드 1, 패딩 0을 적용하여 행렬로 변환
col1 = im2col(x1, 5, 5, stride=1, pad=0)
# 변환된 행렬의 형상 출력
print(col1.shape)

# 0차원(N, 데이터 수)에 10개, 1차원(C, 채널)에 3개, 2차원(H, 데이터 높이)에 7개, 3차원(W, 데이터 너비)에 7개의 0~1 사이 랜덤 실수 원소를 갖는 입력값 행렬 x2
x2 = np.random.rand(10,3,7,7)
# im2col 함수를 이용하여 x2 입력 데이터를 높이 5, 너비 5의 필터, 스트라이드 1, 패딩 0을 적용하여 행렬로 변환
col2 = im2col(x2, 5, 5, stride=1, pad=0)
# 변환된 행렬의 형상 출력
print(col2.shape)

# 데이터 개수(배치)가 1개일 때는 행렬 형상이 (9,75)이지만 데이터 개수(배치)가 10개일 때는 행렬 형상이 10배인 (90,75)


# 반대로 2차원으로 변환된 행렬을 다시 4차원 행렬로 변환하는 col2im 함수 정의(역전파 계산을 위함)
# 변환할 2차원 행렬, 입력값의 형상, 필터값 높이, 필터값 너비, 스트라이드(기본값 1), 패딩(기본값 0) 입력받음
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):

    # 입력값 형상으로부터 데이터의 수, 채널 수, 데이터 높이, 데이터 너비 불러옴
    N, C, H, W = input_shape
    # 출력값 높이 계산
    out_h = (H + 2*pad - filter_h)//stride + 1
    # 출력값 너비 계산
    out_w = (W + 2*pad - filter_w)//stride + 1
    # 행렬값을 (데이터 수, 출력값 높이, 출력값 너비, 채널 수, 필터값 높이, 필터값 너비) 형상의 6차원으로 변환한 후 (0,3,4,5,1,2)순으로 차원을 변형
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    # 출력할 4차원 데이터 형상의 행렬 생성
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    # 필터값 높이만큼 반복
    for y in range(filter_h):
        # y값의 최대값은 y값에 (스트라이드, 출력값 높이)를 더한 값
        y_max = y + stride*out_h
        # 필터값 너비만큼 반복
        for x in range(filter_w):
            # x값의 최대값은 x값에 (스트라이드, 출력값 너비)를 더한 값
            x_max = x + stride*out_w
            # 
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    # 계산된 이미지(4차원 데이터) 행렬 리턴
    return img[:, :, pad:H + pad, pad:W + pad]