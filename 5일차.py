#### 240712 ####
# soft copy
a=[1,2,3]

b=a
b

a[1]=4
a
b

id(a)
id(b)


# deep copy
a=[1,2,3]

b=a[:]
b=a.copy()

a[1]=4
a
b


#수학 함수
x=4

import math
math.sqrt(x)

sqrt_val = math.sqrt(16)
print("16의 제곱근은: ", sqrt_val)

# 수식 계산
x=2
y=9
z=math.pi/2
result = (x**2 + math.sqrt(y) + math.sin(z)) * math.exp(x)
result

def my_f(x,y,z):
  return (x**2 + math.sqrt(y) + math.sin(z)) * math.exp(x)

my_f(2,9,math.pi/2)


def my_f(input):
    contents
    return
  
  
# ch4. 벡터와 친해지기
# ctrl+shift+c : 주석처리
import pandas as pd
import numpy as np

a = np.array([1, 2, 3, 4, 5]) # 숫자형 벡터 생성
b = np.array(["apple", "banana", "orange"]) # 문자형 벡터 생성
c = np.array([True, False, True, True]) # 논리형 벡터 생성
print("Numeric Vector:", a)
print("String Vector:", b)
print("Boolean Vector:", c)

type(a)

a[4]
a[2:]
a[1:4]


x = np.empty(3)
print("빈 벡터 생성하기:", x)

x[0]=3
x[1]=5
x[2]=3
print("채워진 벡터:", x)


vec1=np.array([1,2,3,4,5])
vec1=np.arange(100)
vec1=np.arange(1, 100)
vec1=np.arange(1, 100, 0.5)
vec1

l_space1 = np.linspace(0,1,5) # 일정 간격
l_space1

vec1=np.array([1,2,3,4])
vec1+vec1

max(vec1)
sum(vec1)

# 35672 이하 홀수들의 합은?
vec1=np.arange(1, 35672, 2)
sum(vec1)


# 2차원 배열
b = np.array([[1, 2, 3], [4, 5, 6]])
length = len(b) 
shape = b.shape # 각 차원의 크기
size = b.size # 전체 요소의 개수
length, shape, size

a = np.array([1,2])
b = np.array([1,2,3,4])
a+b


np.tile(a,2)+b  # tile: a를 2번 반복
np.repeat(a,2) +b  # 1,2 -> 1,1,2,2

b==3


# 35672보다 작은 수 중에서 7로 나눠서 나머지가 3인 숫자들의 개수는?
sum((np.arange(1,35672) % 7)==3)
# 10보다 작은 수 중에서 7로 나눠서 나머지가 3인 숫자들의 개수는?
sum((np.arange(1,10) % 7)==3)




#### 240715 ####
# 2차원 배열
matrix = np.array([[0.0, 0.0, 0.0],
 [10.0, 10.0, 10.0],
 [20.0, 20.0, 20.0],
 [30.0, 30.0, 30.0]])
matrix.shape

# 1차원 배열 생성
vector = np.array([1.0, 2.0, 3.0, 4.0])
vector.shape

# 브로드캐스팅을 이용한 배열 덧셈
result = matrix + vector
print("브로드캐스팅 결과:\n", result)


# 세로 벡터 생성
vector = np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1) # 4행1열로 배열바꾸기

# 브로드캐스팅을 이용한 배열 덧셈
result = matrix + vector
print("브로드캐스팅 결과:\n", result)


## 슬라이스
np.random.seed(2024)
a = np.random.randint(1, 21, 10)  # 1~21중에 10개 랜덤 
print(a)

# 두번째 값 추출
print(a[1])

a[2:5]
a[-1] # 끝에서 첫번째
a[::2] # 처음부터 끝까지 2step으로
a[0:6:2]

# 1에서부터 1000 사이 3의 배수의 합은?
sum(np.arange(3, 1001,3))

x = np.arange(1, 1001)
sum(x[2:1000:3]) # x[::3]도 가능


# 두 번째 값 제외하고 추출
a
print(np.delete(a, 1))
np.delete(a,[1,3]) #두 번째 요소와 네 번째 요소를 삭제

b = a[a > 3]
print(b)


np.random.seed(2024)
a = np.random.randint(1, 10000, 5)
a
# 조건을 만족하는 논리형 벡터
a[(a>2000) & (a<5000)]


import pydataset

df = pydataset.data('mtcars')
np_df = np.array(df['mpg'])
np_df

# 15이상 25이하인 데이터 개수는?
sum((np_df>=15) & (np_df<=25))

# 평균 mpg 이상은 몇 대인지?
sum(np_df >= np.mean(np_df))

# 15보다 작거나 22이상인 데이터개수는?
sum((np_df < 15) | (np_df > 22))



np.random.seed(2024)
a = np.random.randint(1, 10000, 5)
b = np.array(["A", "B", "C", "F", "W"])
# a[조건을 만족하는 논리형 벡터]
a[(a>2000) & (a<5000)]
b[(a>2000) & (a<5000)]


model_names = np.array(df.index)
model_names
# 15이상 25이하인 자동차 모델은?
model_names[(np_df>=15) & (np_df<=25)]
# 평균 mpg 이상인 자동차 모델은?
model_names[np_df >= np.mean(np_df)]


a[a>3000] = 3000 # 3000보다 크면 3000으로 바꾸삼
a


np.random.seed(2024)
a = np.random.randint(1, 100, 10)
np.where(a<50) #ture 위치 반환


np.random.seed(2024)
a = np.random.randint(1, 26346, 1000)
# 처음으로 5000보다 큰 숫자는?
a[a>5000][0]
# 그 숫자 위치는?
np.where(a[a>5000][0])
a[np.where(a>5000)][0]

# 다른 예제
# 처음으로 24000보다 큰 숫자와 위치는?
x = np.where(a>24000)
x
my_index = x[0][0]
a[my_index]

# 다른 예제
# 처음으로 10000보다 큰 숫자들 중 50번째 숫자와 위치는?
x = np.where(a>10000)
x
x[0][49]
a[x[0][49]]

# 다른 예제
# 500보다 작은 숫자 중 가장 마지막으로 나오는 숫자 위치와 숫자는?
x = np.where(a<500)
x  #type: 튜플
x[0][-1]  #x[0] type: 넘파이 리스트


 a = np.array([20, np.nan, 13, 24, 309])
 a + 3
np.mean(a)
np.nanmean(a) # nan 무시
np.nan_to_num # nan 값을 다른 값으로 대체
np.nan_to_num(a, nan=0)


# 빈 칸을 제거하는 방법
a_filtered = a[~np.isnan(a)]
a_filtered

# 벡터 합치기
str_vec = np.array(["사과", "배", "수박", "참외"])
str_vec
str_vec[[0, 2]]


mix_vec = np.array(["사과", 12, "수박", "참외"], dtype=str)
mix_vec

combined_vec = np.concatenate((str_vec, mix_vec)) #concatenate: 이어붙이는 함수
combined_vec

col_stacked = np.column_stack((np.arange(1, 5), np.arange(12, 16)))
col_stacked

row_stacked = np.row_stack((np.arange(1, 5), np.arange(12, 16)))
row_stacked
# 이제 vstack 씀
row_stacked = np.vstack((np.arange(1, 5), np.arange(12, 16)))
row_stacked

# 길이가 다른 벡터 합치기
vec1 = np.arange(1, 5)
vec2 = np.arange(12, 18)
vec1 = np.resize(vec1, len(vec2))
vec1

uneven_stacked = np.column_stack((vec1, vec2))
uneven_stacked
uneven_stacked = np.vstack((vec1, vec2))
uneven_stacked


# 연습문제
# 주어진 벡터의 각 요소에 5를 더하기
a = np.array([1,2,3,4,5])
a+5

# 주어진 벡터의 홀수번째 요소만 추출
a = np.array([12, 21, 35, 48, 5])
a[0::2]

# 주어진 벡터에서 최대값
max(a)
a.max()

# 중복된 값을 제거
a = np.array([1, 2, 3, 2, 4, 5, 4, 6])
np.unique(a)

# 주어진 두 벡터의 요소를 번갈아 가면서 합치기
a = np.array([21, 31, 58])
b = np.array([24, 44, 67])
c = np.empty(a.size + b.size, dtype=a.dtype)
c[0::2] = a
c[1::2] = b
c 

x = np.empty(6)
x
x[[1,3,5]] = b #짝수  #.[1::2] 같은말
x[[0,2,4]] = a #홀수

# 다음 a 벡터의 마지막 값은 제외한 두 벡터 a와 b를 더한 결과
a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9])

a[:-1] + b
np.delete(a, -1) + b
