# DeepLearningStudy

딥러닝 공부하면서 실습한 파일 정리하는 곳

Deep Learning from Scratch 밑바닥부터 시작하는 딥러닝(사이토 고키, 한빛 미디어) 책 기반

역시 빡세군

******

### 최근 진행 내용

#### 2022-07-02
**Chapter4**\
4.1 ~ 4.2.1(p.107~p.113)
- 딥러닝을 비롯한 기계학습은 데이터를 이용하여 학습
- 딥러닝은 데이터를 입력받은 후 사람의 개입없이 추정 결과까지의 모든 과정을 컴퓨터가 스스로 학습 및 판단
- 기계학습을 진행할 때 학습 데이터를 '훈련 데이터'와 '시험 데이터'로 이원화하는 과정 필요(특정 데이터에만 편향적이지 않도록 범용성을 가지기 위함)
- 손실함수를 통해 신경망이 추정한 결과와 실제 정답을 비교하여 신경망이 얼마나 정확한지 판정
- 오차제곱합 함수(손실함수) 구현

#### 2022-06-30
**Chapter3**\
3.4 ~ 3.7(p.83~p.106)
- 임의의 가중치와 편향이 정의된 3층 신경망 순방향 처리 구현
- 출력층에서 사용되는 소프트맥스 함수의 개념과 특성
- 기계학습 문제의 두 가지 종류(회귀, 분류)
- 기계학습의 두 가지 단계 (학습(훈련), 추론(분류))
- MNIST 데이터셋을 이용한 손글씨 분류 신경망의 추론과정 구현
- 데이터를 배치(한 묶음)로 한꺼번에 처리

#### 2022-06-28
**Chapter3**\
3.3 ~ 3.3.3(p.77~p.82)
- 다차원 배열(행렬)의 기본 개념
- 행렬곱 계산 방법과 조건
- 행렬곱의 구현

#### 2022-06-27
**Chapter3**\
3 ~ 3.2.7(p.63~p.77)
- 기본적인 신경망 구조(입력층-은닉층-출력층)
- 활성화 함수의 기본 개념 : 어떤 계산 결과가 활성화를 일으키는지(조건을 만족하는지) 판정하여 출력값을 결정
- 3가지 활성화 함수 구현 (계단함수, 시그모이드 함수, ReLU 함수)
- 선형 함수와 비선형 함수 : 신경망에 사용되는 활성화 함수는 비선형 함수를 사용

#### 2022-06-26
Chapter1은 파이썬 기본 문법 및 라이브러리 사용법이라 패스
Chapter2 전체 내용(p.47~p.62)
- 기본적인 퍼셉트론 개념
- 퍼셉트론은 크게 입력값, 가중치, 편향 3가지 기본 요소로 구성
- 기본 퍼셉트론을 이용한 AND 게이트, OR 게이트, NAND 게이트 구현
- 퍼셉트론을 여러개 쌓아서 다층 퍼셉트론을 구현하면 복잡한(비선형) 구조 구현 가능
- 다층 퍼셉트론을 이용한 XOR 게이트 구현
