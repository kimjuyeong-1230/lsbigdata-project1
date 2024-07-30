#### 240722 ####

exit
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "apple", 3.5, True]
print("과일 리스트:", fruits)
print("숫자 리스트:", numbers)
print("혼합 리스트:", mixed)


# 빈 리스트 생성
empty_list1 = []
empty_list2 = list()
print("빈 리스트 1:", empty_list1)
print("빈 리스트 2:", empty_list2)


# 초기값을 가진 리스트 생성
numbers = [1, 2, 3, 4, 5]
range_list = list(range(5))
print("숫자 리스트:", numbers)
print("range() 함수로 생성한 리스트:", range_list)

# 다양한 타입의 리스트 생성
mixed_list = [1, "apple", 3.5, True]
print("혼합 리스트:", mixed_list)


## 리스트 접근 및 슬라이싱
fruits = ["apple", "banana", "cherry", "date", "elderberry"]

# 인덱싱
first_fruit = fruits[0]
last_fruit = fruits[-1]
print("첫 번째 과일:", first_fruit)
print("마지막 과일:", last_fruit)

# 슬라이싱
some_fruits = fruits[1:4]
print("일부 과일:", some_fruits)

# 리스트 원소 수정
fruits = ["apple", "banana", "cherry"]
fruits[1] = "blueberry"
print("수정된 과일 리스트:", fruits)


### 내포
squares = [x**2 for x in range(10)]
print("제곱 리스트:", squares)

# 3, 5, 2, 15의 3제곱
my_squares = [x**3 for x in [3,5,2,15]]
my_squares


# numpy array 와도 가능
import numpy as np
my_squares = [x**3 for x in np.array([3,5,2,15])]
my_squares

# pandas 시리즈 와도 가능?
import pandas as pd
exam = pd.read_csv("data/exam.csv")
my_squares = [x**3 for x in exam["math"]]
my_squares


# 리스트 연결
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined_list = list1 + list2
print("연결된 리스트:", combined_list)

(list1*3) + (list2*5)


numbers = [5,2,3]
repeated_list = [x for x in numbers for _ in range(3)]
repeated_list

## _ 의미
# 1. 앞에 나온 값을 가리킴
5+4
_ + 6 # _는 9를 의미 
# 2. 값 생략 (자리만 나타냄)
a, _, b = (1,2,4)
a;b
_


## for문
for i in range(5):
  print(i**2)

## 리스트 하나 만들고 for루프 사용해서 2,4,6..20의 수를 채워넣어라
# 방법1
mylist = []
for i in range(2,21,2):
  mylist += [i]
mylist

# 방법2
mylist = []
for i in range(2,21,2):
  mylist.append(i)
mylist

# 방법3
mylist = [0] * 10
for i in range(10):
  mylist[i] = 2* (i+1)
mylist

# 다른방법
[i for i in range(2,21,2)]


# 퀴즈: mylist_b의 홀수번째 위치에 있는 숫자들만 mylist에 가져오기 
mylist_b = [2,4,6,80, 10, 12, 24,35,23,20]
mylist = [0] * 5  # 5개의 0으로 초기화된 리스트
for i in range(5):
  mylist[i] = mylist_b[2*i]
mylist 
  

# 리스트 컴프리헨션으로 바꾸는 방법
# 바같은 무조건 대괄호로 묶어줌: 리스트 반환하기 위해서
# for 루프의 : 는 생략한다
# 실행부분을 먼저 써준다
# 결과받는 부분 제외시킴
[i*2 for i in range(1,11)]
[x for x in numbers]


# for문 중첩 (두번째 for문 다 돌아야 첫번째 for문 돌아옴)
for x in numbers:
  for y in range(4):
    print(x)


for i in range(3):
  for j in range(2):
    print(i,j)


for i in [0,1,2]:
  for j in [0,1]:
    print(i,j)
  
  
numbers = [5,2,3]  
for i in numbers:
  for j in range(4):
    print(i)
    
# 리스트 컴프리헨션 변환 
[i for i in numbers for j in range(4)]


# 원소 체크
fruits = ["apple", "banana", "cherry"]
fruits
"banana" in fruits


# [x == "banana" for x in fruits]
mylist = []
for x in fruits:
  mylist.append(x == "banana")
mylist


# 바나나의 위치를 뱉어내게 하려면?
fruits = ["apple", "apple", "banana", "cherry"]

import numpy as np
fruits = np.array(fruits)
np.where(fruits == "banana")[0][0]

# reverse: 원소 순서 뒤에서부터 재배치
fruits.reverse()
fruits

# append: 원소 맨끝에 붙이기
fruits.append("pineapple")

# insert: 원소 삽입
fruits.insert(2, "test")
fruits

# remove: 원소 제거
fruits.remove("apple")
fruits



fruits = np.array(["apple", "banana", "cherry", "apple", "pineapple"])

# 제거할 항목 리스트
items_to_remove = np.array(["banana", "apple"])

# 불리언 마스크 생성
mask = ~np.isin(fruits, items_to_remove)

# 불리언 마스크를 사용하여 항목 제거
filtered_fruits = fruits[mask]
print("remove() 후 배열:", filtered_fruits)




import numpy as np

np.array(33)/sum()/3
(x-16)**2
sum((np.arange(33)-16)**2)  


x = [0,1,2,3]
y = 2/6 + 2* 2/6 + 3*1/6 # 평균: E(x)
E(x^2) = 19/6
y


x = np.arange(4)
x
pro_x = np.array([1/6, 2/6, 2/6, 1/6])
pro_x

#기대값
Ex = sum(x*pro_x)
Exx = sum(x**2 * pro_x)

#분산
Exx - Ex**2

sum((x - Ex)**2 * pro_x)



#### 240724 ####
#!pip install scipy
from scipy.stats import bernoulli
from scipy.stats import binom


# 확률질량함수(pmf)
# 확률변수가 갖는 값에 해당하는 확률을 저장하고 있는 함수
bernoulli.pmf(1, 0.3)  #P(X=1)
bernoulli.pmf(0, 0.3)  #P(X=0)

# P(X =k | n, p)
# n: 베르누리 확률변수 더한 갯수
# p: 1이 나올 확률
# binom.pmf(k, n, p)
binom.pmf(0, n = 2, p = 0.3)
binom.pmf(1, n = 2, p = 0.3)
binom.pmf(2, n = 2, p = 0.3)


## X ~ B(n, p)
# list comp
result = [binom.pmf(x, n=30, p=0.3) for x in range(31)]
result

# numpy
import numpy as np
binom.pmf(np.arange(31), n=30, p=03)


import math
math.factorial(54) / (math.factorial(26) * math.factorial)
math.comb((54, 26))

# 몰라도됨=======================================
# 1*2*3*4
# np.cumprod(np.arnage(1,5))[-1]
# fact_54 = np.cumprod(np.arange(1,55))[-1]
# ln
log(a*b) = log(a) + log(b)
log(1 * 2 * 3 * 4) = log(1) + log(2) + log(3) + log(4)

math.log(24)  #24=1*2*3*4
sum(np.log(np.arange(1,5)))

math.log(math.factorial(54))
logf_54 = sum(np.log(np.arange(1,55)))
logf_26 = sum(np.log(np.arange(1,27)))
logf_28 = sum(np.log(np.arange(1,29)))

# math.comb(54, 26)
np.exp(logf_54 - (logf_26 + logf_28))
# 몰라도됨=======================================


math.comb(2,0)* 0.3**0 * (1-0.3)**2
math.comb(2,1)* 0.3**1 * (1-0.3)**1
math.comb(2,2)* 0.3**2 * (1-0.3)**0

binom.pmf(0, 2, 0.3)
binom.pmf(1, 2, 0.3)
binom.pmf(2, 2, 0.3)


# X~B(n=10, p=0.36)
# P(X=4) = ?
binom.pmf(4, 10, 0.36)

# P(X<=4) = ?
binom.pmf(np.arange(5), n=10, p=0.36).sum()

# P(2 < X <= 8) = ?
binom.pmf(np.arange(3,9), n=10, p=0.36).sum()


# X~B(30, 0.2)
# 확률변수 X가 4보다 작거나 25보다 크거나 같을 확률을 구하시오.
# P(X<=4 or X>=25)
# 1)
a = binom.pmf(np.arange(4), n=30, p=0.2).sum()
# 2)
b = binom.pmf(np.arange(25,31), n=30, p=0.2).sum()
# 3)
a+b
# 4)
1 -  binom.pmf(np.arange(4,25), n=30, p=0.2).sum()


## rvs함수
# 표본 추출 함수
# X1 ~ bernoulli(p=0.3)
bernoulli(p=0.3)
# X2 ~ bernoulli(p=0.3)
bernoulli(p=0.3)
# X~B(n=2, p=0.3)
bernoulli.rvs(0.3) + bernoulli.rvs(0.3)
binom.rvs(n=2, p=0.3, size=1)
binom.pmf(0, n=2, p=0.3)
binom.pmf(1, n=2, p=0.3)


#베르누이 기댓값은 P구나

# X~B(30, p=0.26)
# 표본 30개 뽑아보세요
binom.rvs(n=30, p=0.26, size=30)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 시각화
prob_x = binom.pmf(np.arange(31), n = 30, p=0.26)

sns.barplot(prob_x)
plt.show()
plt.clf()


# 교재 p.207
x = np.arange(31)
prob_x = binom.pmf(np.arange(31), n = 30, p=0.26)

df = pd.DataFrame({"x":x, "prob":prob_x})
df

sns.barplot(data = df, x="x", y="prob")
plt.show()


# cdf: cumlative dist. function
# 누적확률분포 함수
# F_X(x) = P(X <=x)
binom.cdf(4, n=30, p=0.26)
# P(4<X<=18)
binom.cdf(18, n=30, p=0.26) - binom.cdf(4, n=30, p=0.26)

# P(13<X<20)
binom.cdf(19, n=30, p=0.26) - binom.cdf(13, n=30, p=0.26)



import numpy as np
import seaborn as sns

x_1 = binom.rvs(n=30, p=0.26, size=10)
x_1
x = np.arange(31)
prob_x = binom.pmf(x, n=30, p=0.26)
sns.barplot(prob_x, color = "blue")
plt.show()

#add a point at (2,0)
plt.scatter(x_1, np.repeat(0.002,10), color='red', zorder=100, s = 5)
#zorder = 5 : 점의 z순서를 5로 설정
#s=5 : 점의 크기를 5로 설정 


# 기댓값 표현
plt.axvline(x=7.8, color='green', linestyle='--', linewidth=2)

plt.show()
plt.clf()


binom.ppf(0.5, n=30, p=0.26)
binom.ppf(0.7, n=30, p=0.26)
binom.cdf(8, n=30, p=0.26)
binom.cdf(9, n=30, p=0.26)


1/np.sqrt(2*math.pi)
from scipy.stats import norm

norm.pdf(0, loc=0, scale=1)
norm.pdf(5, loc=3, scale=4)


# linspace
k = np.linspace(-5,5,100)
y = norm.pdf(k, loc=0, scale=1)
y

plt.plot(k, y, color="black")
plt.show()
plt.clf()


## mu(loc) : 분포의 중심 결정하는 모수
k = np.linspace(-5,5,100)
y = norm.pdf(k, loc=0, scale=1)  #중심이 3으로 이동
y

plt.plot(k, y, color="black")
plt.show()
plt.clf()


## sigma : 분포의 퍼짐 결정하는 모수(표준편차)
k = np.linspace(-5,5,100)
y = norm.pdf(k, loc=0, scale=1)  
y2 = norm.pdf(k, loc=0, scale=2)
y3 = norm.pdf(k, loc=0, scale=0.5)
plt.plot(k, y, color="black")
plt.plot(k, y2, color="red")
plt.plot(k, y3, color="blue")
plt.show()
plt.clf()
#스케일이 작을수록 뾰족(=표준편차,분산이 작다)


norm.cdf(0, loc=0, scale=1)
norm.cdf(100, loc=0, scale=1)
norm.cdf(0.54, loc=0, scale=1) - norm.cdf(-2, loc=0, scale=1)
norm.cdf(0.54, loc=0, scale=1) - norm.cdf(-2, loc=0, scale=1)


# 정규분포: Normal distribution
# X ~ N(3, 5^2)
# P(3 < X < 5) =? 15.54%
norm.cdf(5,3,5) - norm.cdf(3,3,5)
# 위 확률변수에서 표본 100개를 뽑아보자!
x = norm.rvs(loc=3, scale=5, size=1000)
sum((x > 3) & (x < 5))/1000

# 평균:0, 표준편차:1
# 표본 1000개 뽑아서 0보다 작은 비율 확인
x = norm.rvs(loc=0, scale=1, size=1000)
sum(x < 0)/1000  # np.mean(x<0)


## 표본의 히스토그램을 그린 후, 같은 정규분포의 확률밀도함수(PDF)를 히스토그램 위에 플롯팅
x = norm.rvs(loc=3, scale=2, size=1000) # 표본생성
x
sns.histplot(x, stat = "density")
plt.show()

# 정규분포의 확률밀도함수(PDF) - 선그래프
xmin, xmax = (x.min(), x.max())
x_values = np.linspace(xmin, xmax, 100) # xmin,xmax 사이를 100개의 균등한 간격으로 나눈 값
pdf_values = norm.pdf(x_values, loc=3, scale=2)
plt.plot(x_values,pdf_values, color='red', linewidth=2)
plt.show()
plt.clf()


#### 숙제 qmd
## 1. 정규분포 pdf 값을 계산하는 자신만의 파이썬 함수를 정의하고,
# 정규분포 mu=3, sigma=2의 pdf를 그릴것
def y(a,mean,sigma):
  pdf_values = norm.pdf(a, loc=mean, scale=sigma)
  plt.plot(a, pdf_values, color='red', linewidth=2)
  plt.show()

x_values = np.linspace(-10, 10, 100)
y(x_values,mean=3,sigma=2)


## 2. 파이썬 scipy 패키지 사용해서 다음과 같은 확률을 구하시오
# X ~ N(2, 3^2)
# 1) P(X < 3)
# 2) P(2 < X < 5)
# 3) P(X < 3 or X > 7)

# 1)
x = norm.rvs(loc=2, scale=3, size=1000)
sum(x < 3)/1000
# 2)
x = norm.rvs(loc=2, scale=3, size=1000)
sum((x > 2) & (x < 5))/1000
# 3)
x = norm.rvs(loc=2, scale=3, size=1000)
sum((x < 3) | (x > 7))/1000


## 3. LS빅데이터스쿨 학생들의 중간고사 점수는 평균:30, 분산:4인
# 정규분포를 따른다. 상위 5%에 해당하는 학생의 점수는?
x = norm.rvs(loc=30, scale=2, size=100)
x.sort()
x[-5:]



##### 240726 ####
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
from scipy.stats import norm

## uniform: 균일분포 X ~ U(a,b)
# uniform.pdf(x, loc=0, scale=1)
# uniform.cdf(x, loc=0, scale=1)
# uniform.ppf(q, loc=0, scale=1)
# uniform.rvs(loc=0, scale=1, size=None, random_state)


#loc: 구간시작점(a), scale: 구간길이 (b-a)
uniform.rvs(loc=2, scale=4, size=1)

k = np.linspace(0,8,100)
y = uniform.pdf(k, loc=2, scale=4)
plt.plot(k, y, color= "black")
plt.show()
plt.clf()

# P(X<3.25)
uniform.cdf(3.25, loc=2, scale=4)  #넓이: 1.25*0.25

# P(5<X<8.39)
uniform.cdf(8.39, loc=2, scale=4) - uniform.cdf(5, loc=2, scale=4) # 8.39가 6이어도 괜찮

# 상위 7% 값은?
uniform.ppf(0.93, loc=2, scale=4)


# 표본 20개를 뽑아서 표본평균도 계산해보세요
x = uniform.rvs(loc=2, scale=4, size = 20*1000, random_state=42) #20개씩 천개
x = x.reshape(1000, 20)
x.shape
blue_x = x.mean(axis=1)
blue_x

import seaborn as sns
sns.histplot(blue_x, stat = "density")
plt.show()
plt.clf()


## X bar ~ N(mu, sigma^2/n)
## X bar ~ N(4, 1.33333/20)
uniform.var(loc=2, scale=4)  #분산 #loc:최솟값, scale: 분포의 범위
uniform.expect(loc=2, scale=4) #기댓값


# 정규분포의 확률밀도함수(PDF) - 선그래프
xmin, xmax = (blue_x.min(), blue_x.max())
x_values = np.linspace(xmin, xmax, 100) # xmin,xmax 사이를 100개의 균등한 간격으로 나눈 값
pdf_values = norm.pdf(x_values, loc=4, scale=np.sqrt(1.33333/20))
plt.plot(x_values,pdf_values, color='red', linewidth=2)
plt.show()
plt.clf()


## 신뢰구간
## X bar ~ N(mu, sigma^2/n)
## X bar ~ N(4, 1.33333/20)

# 정규분포의 확률밀도함수(PDF) - 선그래프
x_values = np.linspace(3,5, 100) # xmin,xmax 사이를 100개의 균등한 간격으로 나눈 값
pdf_values = norm.pdf(x_values, loc=4, scale=np.sqrt(1.33333/20))
plt.plot(x_values,pdf_values, color='red', linewidth=2)
plt.show()

# 표본평균(파란벽돌 하나) 점찍기-모평균
blue_x = uniform.rvs(loc=2, scale=4, size = 20).mean()
a = blue_x+0.665
b = blue_x-0.665
plt.scatter(blue_x, 0.002, color="blue", zorder=10, s=10)
plt.axvline(x=a, color="blue", linestyle="--", linewidth=1)
plt.axvline(x=b, color="blue", linestyle="--", linewidth=1)

# 기대값 표현
plt.axvline(x=4, color="green", linestyle="--", linewidth=2)
plt.show()
plt.clf()


# 95%커버 a,b 표준편하 기준 몇배로 벌리면 되나요?
# 약 1.96(양쪽으로)


# 얼마나 떨어져있는가
4 - norm.ppf(0.025, loc=4, scale = np.sqrt(1.33333/20))  
4 - norm.ppf(0.975, loc=4, scale = np.sqrt(1.33333/20))  



#### 240729 ####
import matplotlib.pyplot as plt
import numpy as np

# 점을 직선으로 이어서 표현
k = np.linspace(0,8,100)
y = uniform.pdf(k, loc=2, scale=4)
plt.plot(k, y, color="black")
plt.show()
plt.clf()


# y = x^2을 점 3개 사용해서 그리기
x = np.linspace(-8,8,100)
y = x**2
#plt.scatter(x,y,s=3)
plt.plot(x, y, xlim=(-20,20), color="blue")
plt.show()
plt.clf()

# x축 y축 범위설정
plt.xlim(-10,10)
plt.ylim(0,40)
# 비율 맞추기
plt.gca().set_aspect('equal', adjustable=)
plt.show()
plt.clf()


# p.57
# 작년 남학생 3학년 전체 분포의 표준편차는 6kg이었다고 한다.
# 이 정보를 이번년도 남학생 분포의 표준편차로 대체하여
# 모평균에 대한 90% 신뢰구간을 구하세요.
x = np.array([79.1, 68.8, 62.0, 74.4, 71.0, 60.6, 98.5, 86.4,
              73.0, 40.8, 61.2, 68.7, 61.6, 67.7, 61.7, 66.8])
x.mean()
len(x)

z_005 = norm.ppf(0.95, loc=0, sclae=1)
z_005

#신뢰구간
x.mean() + z_005 * 6 / np.sqrt(16)
x.mean() - z_005 * 6 / np.sqrt(16)


# 데이터로부터 E[X^2] 구하기
x = norm.rvs(loc=3, scale=5, size=100000)

np.mean(x**2)
#sum(x**2) / len(x)-1
np.mean((x - x**2) / (2*x))

np.random.seed(20240729)
x = norm.rvs(loc=3, scale=5, size=100000)
x_bar = x.mean()
s_2 = sum((x - x_bar)**2)/(100000-1)
s_2
#np.mean((x - x_bar)**2)
np.var(x, ddof=1) # n-1로 나눈 값(표본분산)
#np.var(x) 사용하면 안됨 주의!  # n으로 나눈 값 

# n-2 vs n
x=norm.rvs(loc=3, scale=5, size=20)
np.var(x)
np.var(x,ddof=1)
