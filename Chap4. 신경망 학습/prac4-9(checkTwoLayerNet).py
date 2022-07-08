# 구현된 2층 신경망 확인

from TwoLayerNet import TwoLayerNet

# 입력층 뉴런(입력 데이터 크기)이 784개(28×28 픽셀), 은닉층 뉴런이 100개, 출력층 뉴런이 10개((0~9)원-핫 인코딩)인 2층 신경망을 net에 저장
net = TwoLayerNet(input_size=784,hidden_size=100,output_size=10)
# net 신경망의 1층 가중치 매개변수 행렬 형상 출력 (입력층 뉴런 784개 × 은닉층 뉴런 100개)
print(net.params['W1'].shape)
# net 신경망의 1층 편향 매개변수 행렬 형상 출력 (은닉층 뉴런 100개)
print(net.params['b1'].shape)
# net 신경망의 2층 가중치 매개변수 행렬 형상 출력 (은닉층 뉴런 100개 × 출력층 뉴런 10개)
print(net.params['W2'].shape)
# net 신경망의 2층 편향 매개변수 행렬 형상 출력 (출력층 뉴런 10개)
print(net.params['b2'].shape)