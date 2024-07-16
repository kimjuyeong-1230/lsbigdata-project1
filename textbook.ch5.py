#### 240716 ####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 탐색 함수
# head(), tail()
# shape
# info()
# describe()

# 메서드 vs 속성(attribute)

exam = pd.read_csv("data/exam.csv")
exam.head()
exam.tail()
exam.shape
exam.info()
exam.describe()

type(exam)  #판다스 데이터프레임
var=[1,2,3]
type(var)  #리스트
#var.head() #불가능


# 변수명 바꾸기
exam2 = exam.copy()
exam2 = exam2.rename(columns={"nclass" : "class"})


# 파생변수 만들기
exam2["total"] = exam2["math"] + exam2["english"] + exam2["science"]
exam2.head()


exam2["test"] = np.where(exam2["total"] >= 200, "pass", "fail")
exam2.head()

# 막대그래프로 빈도 표현하기
exam2["test"].value_counts().plot.bar(rot=0) #rot=0 : 축 이름 수평
plt.show()

# 현재 그림을 지우기
plt.clf()


exam2["test2"] = np.where(exam2["total"] >= 200, "A",
                np.where(exam2["total"] >= 100, "B", "C"))
exam2.head()

exam2["test2"].isin(['A', 'C'])
