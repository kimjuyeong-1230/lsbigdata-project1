#### 240718 ####
import numpy as np

matrix = np.vstack(
  (np.arange(1,5),
  np.arange(12,16))
)
matrix
type(matrix)
print("행렬: \n", matrix)

np.zeros(5)
type(np.zeros((5,4)))
type(np.zeros([5,4]))

np.arange(1,5).reshape((2,2))  # 2행2열
np.arange(1,5).reshape((2,-1))  # -1: 몇갠지셀필요 x, 알아서 채워줌

# Q. 0에서 부터 99까지 수 중 랜덤하게 50개 숫자 뽑고 5 by 10행렬 만드세요.
np.random.seed(2024)
ran = np.random.randint(0,100,50)
ran.reshape((5,10))


np.arange(1,21).reshape((4,5))  # 가로로 채움(order='T')
mat_a = np.arange(1,21).reshape((4,5), order='F') # 세로로 채움
mat_a
mat_b = np.arange(1,101).reshape((20, -1))
mat_b

# 인덱싱
mat_a[0,0]
mat_a[1,1]
mat_a[2,3]
mat_a[0:2,3]
mat_a[1:3,1:4]
mat_a[3,] # 모든 열
mat_a[3,::2]
mat_b[1::2, :] # 짝수행만
mat_b[[1,4,6,14], :] # 원하는 행만


x = np.arange(1,11).reshape((5,2)) * 2
x[[True, True, False, False, True] ,0]  #ture에 해당하는 행만, 0번째 열만


mat_b[:,1]  # 1차원  # 벡터
mat_b[:,1:2] # 2차원(슬라이스 사용하면 차원 유지)
mat_b[:,1].reshape((-1,1))  # 행렬
mat_b[:,(1,)]  # 행렬


# 필터링
mat_b[mat_b[:,1] % 7 == 0, :]  #두번째 열(mat_b[:,1]) 중 7의 배수 true 값이 있는 행


# 사진은 행렬이다
import matplotlib.pyplot as plt
# 난수 생성하여 3x3 크기의 행렬 생성
np.random.seed(2024)
img1 = np.random.rand(3, 3)
print("이미지 행렬 img1:\n", img1)

plt.imshow(img1, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()


a = np.random.randint(0,10,20).reshape(4,-1)
a/9

x = np.arange(1,11).reshape((5,2)) 
print("원래 행렬\n", x)
x.transpose()


import urllib.request


#!pip install imageio
import imageio

# 이미지읽기
jelly = imageio.imread("C:/Users/USER/Documents/LS빅데이터스쿨/lsbigdata-project1/img/jelly.png")


## 3차원 배열
# 두개의 2x3 행렬 생성
mat1 = np.arange(1,7).reshape(2,3)
mat2 = np.arange(7,13).reshape(2,3)

my_array = np.array([mat1, mat2])
my_array
my_array.shape

my_array2 = np.array([my_array,my_array])
my_array2
my_array2.shape
my_array2[0,:,:,:]

my_array[:,0,:]
my_array[0,1,1:] # 5,6만


mat_x = np.arange(1,101).reshape((5,5,4))
mat_x
mat_x = np.arange(1,101).reshape((10,5,2))
mat_x
mat_x = np.arange(1,100).reshape((-1,3,3))
mat_x
len(mat_x)



# 넘파이 배열 메서드
a = np.array([[1,2,3], [4,5,6]])
a.sum()
a.sum(axis=0)
a.sum(axis=1)

a.mean()
a.mean(axis=0)
a.mean(axis=1)


mat_b = np.random.randint(0,100,50).reshape((5,-1))
mat_b

# 가장 큰 수는?
mat_b.max()
# 행별로 가장 큰 수는?
mat_b.max(axis=1)
# 열별로 가장 큰 수는?
mat_b.max(axis=0)


a = np.array([1,3,2,5])
a.cumsum()  #누적합
a.cumprod() #누적곱

mat_b.cumsum(axis=1)
mat_b.cumprod(axis=1)


mat_b.reshape((2,5,5))
mat_b.flatten() # 다차원을 벡터형태로(1차원으로)


d = np.array([1,2,3,4,5])
d.clip(2,4) # 2를 최소값으로 4를 최대값으로 되게 하라(범위밖숫자를 경계숫자로)


d.tolist()  # 리스트 형태



### 강사님 github > code > stat1에 코드 올라와있음
## 균일확률변수 만들기
np.random.rand(1)

def X(i):
  return np.random.rand(i)

X()


## 베르누이 확률변수 모수: p 만들어보세요
def Y(num, p):
  x = np.random.rand(num)
  return np.where(x < p, 1, 0)

Y(3, 0.5)  # Y(num=3, p=0.5)
sum(Y(3, 0.5))/3 #평균 
Y(3, 0.5).mean() #평균

Y(10000, 0.5).mean()


# 새로운 확률변수
# 가질 수 있는 값 : 0,1,2
# 20%, 50%, 30%
def Z():
  x = np.random.rand(1)
  return np.where(x<0.2, 0, np.where(x<0.7,1,2))

Z()


p = np.array([0.2, 0.5, 0.3])
def Z(p):
  x = np.random.rand(1)
  p_cumsum = p.cumsum()
  return np.where(x<p_cumsum[0], 0, np.where(x<p_cumsum[1],1,2))

p = np.array([0.2, 0.5, 0.3])
Z(p)


#### 240719 ####
import matplotlib.pyplot as plt
# 0과 1사이 숫자 5개 발생
x = np.random.rand(10000).reshape(-1,5).mean(axis=1)
plt.hist(x, bins=4, alpha=0.7, color='blue)
plt.title('Histogram of Numpy Value")
plt.xlavel('Vector')
plt.ylavel('Histogram of Numpy Vector')
plt.grid(False)
plt.show()


 x = np.random.rand(10000,5).mean(axis=1)
 
