#### 20240801 ####
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

# 직선의 방정식
# y = ax + b

# 예시: y = ax + 3의 그래프를 그려보세요!
a=80
b=5  #숫자가 커질수록 위로 올라감

x = np.linspace(0,5,100)
y = a*x + b

house_train = pd.read_csv("./data/houseprice/train.csv")
house_train

my_df = house_train[["BedroomAbvGr", "SalePrice"]].head(10)
my_df["SalePrice"] = my_df["SalePrice"] /1000
plt.scatter(x=my_df["BedroomAbvGr"], y=my_df["SalePrice"])
plt.plot(x,y,color = "blue")
plt.show()
plt.clf()



## 점에 가까이 선그래프 맞추기 
# 곡선인 경우
#a = 5
#b = 100
#x = np.linspace(0,5, 100)
#y = np.exp(x) * a + b

# 직선인 경우
a1 = 80
b1 = -30
y = a1 * x + b1

my_df = house_train[["BedroomAbvGr", "SalePrice"]].head(10)
my_df["SalePrice"] = my_df["SalePrice"] /1000
plt.scatter(x=my_df["BedroomAbvGr"], y=my_df["SalePrice"], color="orange")
plt.plot(x,y,color = "blue")
plt.ylim(-1, 400)
plt.show()
plt.clf()


## 점수 맞추기(서브미션)
test = pd.read_csv("data/houseprice/test.csv")
test = test.assign(SalePrice = ((80* test["BedroomAbvGr"] -30) * 1000))
test["SalePrice"]

sample_submission = pd.read_csv("data/houseprice/sample_submission.csv")
sample_submission["SalePrice"] = test["SalePrice"]
# sample_submission.to_csv("sample_submission_240801.csv", index = False)


## 직선 성능 평가
a = 80
b = -60

# y_hat은 어떻게 구할까?
y_hat = (a * house_train["BedroomAbvGr"] + b) * 1000
# y는 어디에 있는가?
y = house_train["SalePrice"]

np.abs(y-y_hat) # 절대거리
np.sum(np.abs(y-y_hat))


## 한글
from matplotlib import font_manager, rc

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # 예시: 윈도우 시스템에 있는 맑은 고딕 폰트 경로
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)


### 회귀분석 직선 구하기 ###
# !pip install scikit-learn
from sklearn.linear_model import LinearRegression

## 개념 (연습)
# 예시 데이터 (x와 y 벡터)
x = np.array([1, 3, 2, 1, 5]).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = np.array([1, 2, 3, 4, 5])  # y 벡터 (레이블 벡터는 1차원 배열입니다)

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_  # 기울기 a
model.intercept_  # 절편 b

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='실제 데이터')
plt.plot(x, y_pred, color='red', label='회귀 직선')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()



## house 데이터 사용 ##
# 데이터 불러오기
train = pd.read_csv("data/houseprice/train.csv")
test = pd.read_csv("data/houseprice/test.csv")
sample_submission = pd.read_csv("data/houseprice/sample_submission.csv")

# 회귀분석 적합(fit)하기
x = np.array(house_train["BedroomAbvGr"]).reshape(-1, 1) #시리즈에선 reshape이 안됨
y = np.array(house_train["SalePrice"]) / 1000
#> x: 방 개수, y: 집값

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_  # 기울기 a
#>16.38101698
model.intercept_  # 절편 b
#>133.96602049739172

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")



# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='실제 데이터')
plt.plot(x, y_pred, color='red', label='회귀 직선')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()


# 직선 성능 평가
a = 16.38101698
b = 133.96602049739172

# y_hat은 어떻게 구할까?
y_hat = (a * house_train["BedroomAbvGr"] + b) * 1000
# y는 어디에 있는가?
y = house_train["SalePrice"]

np.abs(y-y_hat) # 절대거리
np.sum(np.abs(y-y_hat))


from scipy.optimize import minimize
def my_f(x):
  return x**2 + 3

my_f(3)

# 초기 추정값
initial_guess = [0]

# 최소값 찾기
result = minimize(my_f, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x값", result.x)




#### 240802 ####
# f(x,y) = (x-1)^2 + (y-3)^2 + 3
def my_f2(x):
  return x[0]**2 + x[1]**2 + 3

my_f2([1,3])

# 초기 추정값
initial_guess = [-10,3]

# 최소값 찾기
result = minimize(my_f2, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x값", result.x)


# f(x,y,z) = (x-1)^2 + (y-2)^2 + (z-4)^2 + 7
def my_f3(x):
  return (x[0]-1)**2 + (x[1]-2)**2 + (x[2]-4)**2 + 3


# 초기 추정값
initial_guess = [-10,3,4]

# 최소값 찾기
result = minimize(my_f3, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x값", result.x)




## 원하는 변수 만들기1
house_train = pd.read_csv("data/houseprice/train.csv")
house_test = pd.read_csv("data/houseprice/test.csv")
sample_submission = pd.read_csv("data/houseprice/sample_submission.csv")

# 회귀분석 적합(fit)하기
x = np.array(house_train["LotArea"]).reshape(-1, 1) #시리즈에선 reshape이 안됨
y = np.array(house_train["SalePrice"]) / 1000
#> x: 방 개수, y: 집값

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_  # 기울기 a
#>16.38101698
model.intercept_  # 절편 b
#>133.96602049739172

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")



# 예측값 계산
y_pred = model.predict(x)
y_pred


house_test = house_test.assign(SalePrice = ((slope* test["LotArea"] +intercept) * 1000))
house_test["SalePrice"]


sample_submission["SalePrice"] = house_test["SalePrice"]
sample_submission

sample_submission.to_csv("sample_submission6.csv", index = False)


## 원하는 변수 만들기2 (이상치제거)
# 회귀분석 적합(fit)하기
x = np.array(house_train["GrLivArea"]).reshape(-1, 1) #시리즈에선 reshape이 안됨
y = np.array(house_train["SalePrice"]) / 1000
#> x: 방 개수, y: 집값

# 이상치탐색
house_train = house_train.query("GrLivArea <= 4500")

x = np.array(house_train["GrLivArea"]).reshape(-1, 1) #시리즈에선 reshape이 안됨
y = np.array(house_train["SalePrice"])

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_  # 기울기 a
model.intercept_  # 절편 b

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")


# 예측값 계산
y_pred = model.predict(x)
y_pred

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()


house_test = house_test.assign(SalePrice = ((slope* house_test["GrLivArea"] +intercept) * 1000))
house_test["SalePrice"]


sample_submission["SalePrice"] = house_test["SalePrice"]
sample_submission

sample_submission.to_csv("sample_submission7_3.csv", index = False)



#### 간단 방법
train_x = np.array(house_train["GrLivArea"]).reshape(-1, 1) #시리즈에선 reshape이 안됨
train_y = np.array(house_train["SalePrice"])

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(train_x, train_y) # 자동으로 기울기, 절편 값을 구해줌

# 예측값 계산
y_pred_train = model.predict(train_x)
y_pred_train


test_x = np.array(house_test["GrLivArea"]).reshape(-1, 1) #시리즈에선 reshape이 안됨

# 예측값 계산
y_pred_test = model.predict(test_x)
y_pred_test


sample_submission["SalePrice"] = y_pred_test
sample_submission

sample_submission.to_csv("sample_submission7_1.csv", index = False)



## 원하는 변수 사용하기 (변수 2개 사용)
x = house_train[["GrLivArea", "GarageArea"]]
y = np.array(house_train["SalePrice"])

# 이상치탐색
house_train = house_train.query("GrLivArea <= 4500")

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_  # 기울기 a
model.intercept_  # 절편 b

# 함수 만들기
def my_f3(x,y):
  return model.coef_[0] * x + model.coef_[1] * y + model.intercept_

my_f3(house_test["GrLivArea"],house_test["GarageArea"])


test_x = house_test[["GrLivArea", "GarageArea"]]
test_x

# 결측치 확인
test_x["GrLivArea"].isna().sum()
test_x["GarageArea"].isna().sum()

# 결측치 대체
test_x= test_x.fillna(house_test["GarageArea"].mean())
test_x

pred_y = model.predict(test_x)
pred_y

sample_submission["SalePrice"] = pred_y
sample_submission

sample_submission.to_csv("sample_submission8_1.csv", index = False)


