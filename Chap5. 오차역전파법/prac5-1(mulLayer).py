# 오차역전파법(역전파법) : 가중치 매개변수의 기울기를 효율적으로 구하는 방법

# 곱셉 노드 구현
# 곱셈 노드는 순전파에서 2개의 입력값(x,y)을 곱한 값(out)을 출력하는 노드
# 역전파에서는 상류의 값(앞의 노드에서 흘러온 값)에서 순전파 때의 입력신호 중 자신이 아닌 상대 신호의 값을 곱한 값을 출력

# 곱셈 노드 클래스 정의
class MulLayer:
    # 생성자에서 입력값 변수(self.x, self.y)를 미리 생성
    def __init__(self):
        self.x = None
        self.y = None

    # 순전파 구현 : 입력값 2개(x,y)를 입력하면 입력값들을 (역전파 계산에 사용하기 위해)클래스 변수(self.x, self.y)로 저장한 후 곱한 값(out) 출력
    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x*y
        
        return out

    # 역전파 구현 : 상류에서 넘어온 미분값(dout)에 순전파 계산에서 저장된 입력값을 서로 바꾸어 곱한 후 해당되는 하류 방면으로 출력
    def backward(self,dout):
        dx = dout*self.y
        dy = dout*self.x
        return dx, dy


# 순전파와 역전파를 이용하여 사과 쇼핑 구현하기
# 100원짜리 사과를 2개 구입하여 소비세가 10% 부과될 때의 총 계산 금액

# 순전파에서의 입력값 : 사과 금액 100, 사과 개수 2, 소비세 1.1(10%를 추가하는 것이므로) 3개
apple = 100
apple_num = 2
tax = 1.1

# 순전파에서의 노드(계층) : 사과 금액과 사과 개수를 곱하는 곱셈 노드(계층), 총 사과 금액과 소비세를 곱하는 곱셈 노드(계층) 2개
# 각 계층을 곱셈 노드 클래스로 저장
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# 곱셈 노드 클래스를 통해 순전파 계산을 하고 이 때 사용된 입력값을 클래스 내부 변수로 저장(추후 역전파 계산에 사용하기 위해)
# 사과 가격(apple)과 사과 구매 개수(apple_num)를 곱셈 노드 순전파(그냥 곱)로 계산하여 사과 총액(apple_price)으로 저장
apple_price = mul_apple_layer.forward(apple, apple_num)
# 사과 총액(apple_price)와 소비세(tax)를 곱셈 노드 순전파로 계산하여 계산 총액(price)으로 저장
price = mul_tax_layer.forward(apple_price, tax)


# 최종 순전파 계산 결과(계산 총액) 출력
print(price)


# 이 계산 식에서 각 변수(사과 총액, 소비세, 사과 가격, 사과 개수)에 대한 미분을 역전파를 통해 계산
# 역전파에서는 순전파에서의 출력값에 대한 미분값을 입력받아 각 변수에 대한 미분값을 각각 출력

# 출력값에 대한 미분값 dprice 정의 (상수를 미분하면 그냥 1)
dprice = 1
# 사과 계산 식에서 사과 총액에 대한 미분값, 소비세에 대한 미분값을 계산 총액 계산 계층에서의 역전파에 순전파 출력값(계산 총액)을 미분한 값을 입력하여 계산
dapple_price, dtax = mul_tax_layer.backward(dprice)
# 사과 계산 식에서 사과 가격에 대한 미분값, 사과 개수에 대한 미분값을 사과 총액 계산 계층에서의 역전파에 순전파 출력값(사과 총액)을 미분한 값을 입력하여 계산
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

# 계산된 각 순전파 입력값(사과 가격, 사과 개수, 소비세)에서의 미분값 출력
print(dapple, dapple_num, dtax)