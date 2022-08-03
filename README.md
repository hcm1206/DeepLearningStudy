# DeepLearningStudy

딥러닝 공부하면서 실습한 파일 정리하는 곳

Deep Learning from Scratch 밑바닥부터 시작하는 딥러닝(사이토 고키, 한빛 미디어) 책 기반

현재 저의 수듄상 이해하기 난해한 부분이 많아 진도가 느려지는 중입니다

******

*파이썬 실행 환경이 디렉토리 경로를 제대로 인식하지 못하여 sys.path.append(os.pardir) 부분을 절대 경로로 지정해놓은 상태*

*(다른 환경에서 실행시 해당 부분 수정 필요)*


******

### 최근 진행 내용

#### 2022-08-03
**Chapter7**\
7.4.3 ~ 7.4.4(p.245~p.250)
- 구현한 im2col 함수 계산 검증
- 계산에 사용된 행렬을 다시 3차원 이상의 다차원 행렬로 변환하는 col2im 함수 구현
- 합성곱 계층의 순전파와 역전파 구현
- 풀링 계층 : 데이터의 특정 영역에서 원하는 값(최대값, 평균값 등)만을 뽑아내어 크기가 줄어든 별도의 행렬을 만드는 계층
- 풀링 계층의 순전파와 역전파 구현

#### 2022-08-02
**Chapter7**\
7 ~ 7.4.2(p.227~p.244)
- 기존 Affine 계층을 사용하는 신경망은 입력값의 형상을 무시해버리고 1차원으로 계산하는 문제 존재
- 이미지, 음성 인식 등 3차원 이상의 데이터를 신경망에 사용할 때는 합성곱 신경망(CNN) 사용
- 다양한 상황에서의 합성곱 연산법
- 합성곱 연산에서의 고유 요소 : 패딩, 스트라이드
- 배치 처리를 위하여 합성곱 신경망에서는 4차원 데이터를 이용한 합성곱 사용
- 다차원 데이터를 2차원으로 전개하는 im2col 함수 구현

#### 2022-08-01
**Chapter6**\
6.5.2 ~ 6.6(p.223~p.226)
- 하이퍼파라미터 최적화는 최적값이 위치하는 특정한 범위를 찾아서 범위를 줄여나가는 과정
- 하이퍼파라미터의 범위는 주로 10의 거듭제곱 단위로 변경하는 로그 스케일로 지정
- MNIST 데이터셋 신경망에서 학습률과 가중치 감소 계수를 최적화하는 과정 구현

#### 2022-07-31
**Chapter6**\
6.4.3 ~ 6.5.1(p.219~p.223)
- 드롭아웃 : 신경망 학습 시 은닉층의 일부 노드를 비활성화하며 학습을 진행하여 오버피팅을 억제하는 방법
- MNIST 신경망에서 드롭아웃을 사용하여 학습한 결과 테스트
- 신경망 학습에서 사용자가 하이퍼파라미터 값을 적절하게 설정하는 것이 학습 진행에 매우 중요
- 하이퍼파라미터 값을 시험하기 위해서는 학습 데이터, 시험 데이터와 구별된 검증 데이터가 필요
- MNIST 데이터셋의 학습 데이터 중 일부를 검증 데이터로 추출하는 과정 구현

#### 2022-07-30
**Chapter6**\
6.4.2(p.217~p.218)
- 가중치 감소 : 신경망 학습 과정에서 큰 가중치값에 대해 큰 패널티를 부과하여 오버피팅을 억제하는 방법
- MNIST 신경망에서 가중치 감소를 사용하여 학습한 결과 테스트

#### 2022-07-29
**Chapter6**\
6.4 ~ 6.4.1(p.215~p.217)
- 오버피팅 : 신경망이 훈련된 데이터에만 지나치게 적용하여 훈련 데이터 외의 데이터에 대해서 낮은 성능을 보이는 현상
- 오버피팅을 일으키는 2가지 요소 : 복잡한 네트워크, 적은 훈련 데이터
- MNIST 신경망의 은닉층을 7개, 학습 데이터를 300개로 설정하여 오버피팅 발생 실험




