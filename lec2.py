#### 240710
# shift+화살표: 블록

a=1
a

# 파워쉘 명령어 리스트
# ls: 파일목록
# cd: 폴더이동
# .  현재폴더
# .. 상위폴더


# show folder in new window: 해당위치 탐색기

# tab\shift tab : 자동완성, 옵션변경
# cls: 화면정리



a= 10
a

a = "안녕하세요"
a
a = '안녕하세요'
a

a = [1,2,3]
a
b = [4,5,6]
b
a+b


a = "안녕하세요"
a
b = " LS 빅데이터 스쿨"
a+b

print(a)


a = 10
b = 3.3

print("a + b =", a+b)
print("a - b =", a-b)
print("a * b =", a*b)
print("a / b =", a/b)
print("a // b =", a//b)
print("a ** b =", a**b)
# shift + alt + 아래화살표: 아래로 복사
# ctrl + alt + 아래화살표: 커서 여러개


a==b
a!=b
a>b
a<b


# 2에 4승과 12453을 7로 나눈 몫을 더해서 8로 나눴을 때 나머지
(2**4 + 12453//7) % 8
# 9의 7승을 12로 나누고, 36452를 253로 나눈 나머지에 곱한 수
((9**7) / 12) * (36452%253)
# 더 큰 것은?
(2**4 + 12453//7) % 8 < ((9**7) / 12) * (36452%253)


# 사용자 나이 검증 예제
user_age = 25
is_adult = user_age >= 18
print("성인입니까?", is_adult)



TRUE = 4
 a = "True"
 b = TRUE
 c = true
 d = True
 
# 논리연산자
a = True
b = False

a and b
a or b
not a
not b


True * False
True * True
False * False

# or 연산자
True or True
True or False
False or True
False or False


a = True
b = False
min(a + b, 1)

a = False
b = False
min(a + b, 1)


a = 3
a += 10  # a = a + 10
a

a -= 4
a

a %= 3
a

a += 12
a

a **= 2
a
a /= 7
a


str1 = "hello"
str1 + str1
str1 * 3


# 단항연산자
x = 5
+x
-x
~x

# binary: 이진수로 표현
bin(5)
bin(-5)

x = -3
bin(x)
bin(~x)
bin(~5)


max(3,4)
var1 = [1,2,3]
sum(var1)


!pip install pydataset
import pydataset
pydataset.data()

df = pydataset.data("AirPassengers")
df
