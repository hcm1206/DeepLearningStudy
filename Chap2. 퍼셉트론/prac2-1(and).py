# 기본 퍼셉트론을 이용한 AND 게이트 예

# x1, x2는 입력값(inputs)
# w1, w2는 가중치(weights)
# theta는 임계값(θ)

def AND(x1, x2):
    # 가중치와 임계값을 임의로 설정
    w1, w2, theta = 0.5, 0.5, 0.7
    # 입력 값에 가중치를 곱하여 모두 저장한 값을 tmp에 저장
    tmp = x1*w1 + x2*w2
    # tmp가 임계값을 넘으면 1, 아니면 0 출력
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

# AND 게이트에 모든 경우의 수를 입력하여 결과 출력
print(AND(0,0))
print(AND(0,1))
print(AND(1,0))
print(AND(1,1))