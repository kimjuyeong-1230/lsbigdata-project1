# 데이터타입
x = 15
print(x, "는 ", type(x), "형식입니다.", sep='') #sep: 입력값사이에 넣을 거


# 문자형 데이터 예제
a = "Hello, world!"
b = 'python programming'

# 여러 줄 문자열
ml_str = """This is
a multi-line
string"""
print(a, type(a))
print(b, type(b))
print(ml_str, type(ml_str))


# 문자열 결합
greeting = "안녕" + " " + "파이썬!"
print("결합 된 문자열:", greeting)

# 문자열 반복
laugh = "하" * 3
print("반복 문자열:", laugh)


# 리스트 생성 예제
fruits = ['apple', 'banana', 'cherry']
numbers = [1, 2, 3, 4, 5]
mixed_list = [1, "Hello", [1, 2, 3]]
print("Fruits:", fruits)
print("Numbers:", numbers)
print("Mixed List:", mixed_list)

a = (10,20,30)
b_int = (42)
b_int
b_tp=(42,)
b_tp

type(a)
type(b)


# 인덱싱
print("첫번째 좌표:", a[0])
# 슬라이싱
print("마지막 두개 좌표:", a[1:])

a_list = [10,20,30]
a_list[1] = 25

a_tp = (10,20,30,40,50)
a_tp[3:] # 네번째부터 끝까지
a_tp[:3] # 처음부터 세번째까지
a_tp[1:3] #두번째부터 세번째까지


# 사용자 정의함수
def min_max(numbers):
 return min(numbers), max(numbers)

a = [1,2,3,4,5]
result = min_max(a)
result
print("Minimum and maximum:", result)


# 딕셔너리
person = {
 'name': '김주영',
 'age': (24,22),
 'city': ['대전', '서산']
}
print("Person:", person)

person.get('name')
person.get('age')[0]

# 집합
# 중복된 값 필터링, 중괄호 사용, 순서 지맘대로 나열
fruits = {'apple', 'banana', 'cherry', 'apple'}
print("Fruits set:", fruits) # 중복 'apple'은 제거됨

# 빈 집합 생성
empty_set = set()
print("Empty set:", empty_set)

empty_set.add('apple')
empty_set.add('banana')
empty_set.remove('banana') #요소가 집합에 없으면 에러 발생
empty_set.discard('banana') #요소가 집합에 없어도 에러X


# 집합 간 연산
other_fruits = {'berry', 'cherry'}
# 교집합
union_fruits = fruits.union(other_fruits) 
print("Union of fruits:", union_fruits)
# 합집합
intersection_fruits = fruits.intersection(other_fruits)
print("Intersection of fruits:", intersection_fruits)


# 논리형 데이터 예제
p = True
q = False
print(p, type(p))
print(q, type(q))
print(p + p) # True는 1로, False는 0으로 계산

is_active = True
is_greater = 10 > 5 # True 반환
is_equal = (10 == 5) # False 반환
print("Is active:", is_active)
print("Is 10 greater than 5?:", is_greater)
print("Is 10 equal to 5?:", is_equal)


# 조건문
a=3
if (a == 2):
 print("a는 2와 같습니다.")
else:
  print("a는 2와 같지 않습니다.")


# 숫자형을 문자열형으로 변환
num = 123
str_num = str(num)
print("문자열:", str_num, type(str_num))

# 문자열형을 숫자형(실수)으로 변환
num_again = float(str_num)
print("숫자형:", num_again, type(num_again))


# 리스트와 튜플 변환
lst = [1, 2, 3]
print("리스트:", lst)
tup = tuple(lst)
print("튜플:", tup)
type(tup)


set_example = {'a', 'b', 'c'}
# 딕셔너리로 변환 시, 일반적으로 집합 요소를 키 또는 값으로 사용
dict_from_set = {key: True for key in set_example}
print("Dictionary from set:", dict_from_set)



# 교재 63pg
!pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt

var = ['a', 'a', 'b', 'c']
var
sns.countplot(x=var)
plt.show()


## seaborn의 titanic 데이터로 그래프 만들기
df = sns.load_dataset('titanic')
df
sns.countplot(data=df, x='sex')
#plt.clf() #그림 지우기
plt.show()

#plt.clf() #그림 지우기
sns.countplot(data=df, x='class')
plt.show()

?sns.countplot() #함수 설명
sns.countplot(data=df, x='class', hue="alive", color="red")
plt.show()

sns.countplot(data=df, x='sex',hue="sex") #hue:색상구분
plt.show()

sns.countplot(data=df,
              x='class',
              hue="alive",
              orient="v") #orient="v":그래프의 방향 수직
plt.show()


## 모듈 알아보기
!pip install scikit-learn
import sklearn.metrics

from sklearn import metrics #모듈명.함수명()으로 함수 사용하기
metrics.accuracy_score()

from sklearn.metrics import accuracy_score #함수명()으로 함수 사용하기
accuracy_score

import sklearn.metrics as met
