# DeepLearningStudy

딥러닝 공부하면서 실습한 파일 정리하는 곳

Deep Learning from Scratch 밑바닥부터 시작하는 딥러닝(사이토 고키, 한빛 미디어) 책 기반

역시 빡세군

******

### 최근 진행 내용

#### 2022-07-05
**Chapter4**\
4.3.3(p.125~p.127)
- 편미분 : 입력값(변수)이 2개 이상인 함수에 대한 미분
- 편미분 미분 구현(목표 변수 이외의 변수를 상수(고정된 값)로 바꾸어 변수가 하나인 함수를 만들고 그 함수를 미분)

#### 2022-07-03
**Chapter4**\
4.2.2 ~ 4.3.2(p.113~p.125)
- 교차 엔트로피 오차 함수(손실함수) 구현
- 미니배치 학습 : 방대한 양의 데이터들을 학습해야 할 때 데이터들의 일부(미니배치)를 무작위로 골라 학습하는 방법
- 데이터를 미니배치(단일 데이터가 아닌 일정 데이터 묶음)로 입력받을 때의 손실함수 구현
- 손실함수(그리고 활성화함수로 소프트맥스 함수)를 사용하는 이유는 매개변수의 사소한 변화로 인한 결과 값의 사소한 변화를 포착하여 학습 성능을 높이기 위함
- 수치 미분의 기본적인 개념 및 구현

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




