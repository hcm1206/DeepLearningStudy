# 덧셈 노드 구현
# 덧셈 노드는 순전파에서 2개의 입력값(x, y)를 더한 값(out)을 출력하는 노드
# 역전파에서는 상류의 값(dout)에 1을 곱하여(값을 변경하지 않고 그대로) 각 순전파 때의 입력 변수 방향으로 보내어 출력

# 곱셈 노드 클래스

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x*y
        return out

    def backward(self,dout):
        dx = dout*self.y
        dy = dout*self.x
        return dx, dy

# 덧셈 노드 클래스 정의
class AddLayer:
    # 덧셈 노드에서는 순전파에서 사용한 변수를 역전파 계산에 사용하지 않으므로 변수를 따로 저장하지 않음
    def __init__(self):
        pass
    
    #  덧셈 노드 순전파 구현 : 입력값 2개(x,y)를 입력하면 입력값들을 더한 값(out)을 출력
    def forward(self, x, y):
        out = x + y
        return out

    # 덧셈 노드 역전파 구현 : 상류에서 넘어온 미분값(dout)을 각 순전파 입력값 방향 하류로 그대로 보내 출력
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


# 순전파와 역전파를 이용하여 과일 쇼핑 구현
# 100원짜리 사과 2개와 150원짜리 귤 3개를 구입하여 소비세가 10% 부과될 때의 총 계산 금액

# 순전파에서의 입력값 : 사과 금액 100, 사과 개수 2, 귤 금액 150, 귤 개수 3, 소비세 1.1 5개
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# 순전파에서의 계층 : 사과 금액과 사과 가격을 곱하는 곱셈 계층, 귤 금액과 귤 가격을 곱하는 곱셈 계층, 
# 사과 총액과 귤 총액을 더하는 덧셈 계층, 과일 총액과 소비세를 곱하는 곱셈 계층 4개
# 각 계층을 노드 클래스로 저장

# 순전파에서 사과 가격과 사과 개수를 곱하여 사과 총액을 구하는 곱셈 계층
mul_apple_layer = MulLayer()
# 순전파에서 귤 가격과 귤 개수를 곱하여 귤 총액을 구하는 곱셈 계층
mul_orange_layer = MulLayer()
# 순전파에서 사과 총액과 귤 총액을 더하는 덧셈 계층
add_apple_orange_layer = AddLayer()
# 과일 총액과 소비세를 곱하는 곱셈 계층
mul_tax_layer = MulLayer()

# 순전파를 통해 각 변수 계산
# 사과 가격과 사과 개수를 곱하여 사과 총액 계산
apple_price = mul_apple_layer.forward(apple, apple_num)
# 귤 가격과 귤 개수를 곱하여 귤 총액 계산
orange_price = mul_orange_layer.forward(orange, orange_num)
# 사과 총액과 귤 총액을 더하여 과일 총액 계산
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
# 과일 총액과 소비세를 곱하여 계산 총액 계산
price = mul_tax_layer.forward(all_price, tax)


# 역전파를 통해 전체 식에서 각 변수의 미분값 계산
# 전체 식에서 출력값(계산 총액)에 대한 미분값 dprice 정의(상수를 미분하면 그냥 1)
dprice = 1
# 전체 식에서 과일 총엑에 대한 미분값과 소비세에 대한 미분값을 계산 총액 계층에서의 역전파 계산으로 구함
dall_price, dtax = mul_tax_layer.backward(dprice)
# 전체 식에서 사과 총액에 대한 미분값과 귤에 대한 미분값을 과일 총액 계층에서의 역전파 계산으로 구함
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
# 전체 식에서 귤 가격에 대한 미분값과 귤 개수에 대한 미분값을 귤 총액 계층에서의 역전파 계산으로 구함
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
# 전체 식에서 사과 가격에 대한 미분값과 사과 개수에 대한 미분값을 사과 총액 계층에서의 역전파 계산으로 구함
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

# 순전파 계산으로 구한 계산 총액 출력
print(price)
# 역전파 계산으로 구한 사과 개수에 대한 미분값, 사과 가격에 대한 미분값, 귤 개수에 대한 미분값, 귤 가격에 대한 미분값, 소비세에 대한 미분값 출력
print(dapple_num, dapple, dorange, dorange_num, dtax)