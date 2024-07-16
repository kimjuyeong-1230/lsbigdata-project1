,### 240715 ####
import pandas as pd
import numpy as np

df = pd.DataFrame({'name' : ['김지훈', '이유진', '박동현', '김민지'],
'english': [90, 80, 60, 70],
'math': [50, 60, 100, 20]
})

df
df["name"]
type(df)  # 판다스 데이터프레임
type(df[["name"]]) # 판다스 데이터프레임
type(df["name"]) # 판다스 시리즈

sum(df["english"])/4

# 문제
df = pd.DataFrame({'제품' : ['사과', '딸기', '수박'],
'가격': [1800, 1500, 3000],
'판매량': [24, 38, 13]
})
df

sum(df['가격'])/3
sum(df['판매량'])/3


!pip install openpyxl
import pandas as pd
df_exam = pd.read_excel("data/excel_exam.xlsx")
df_exam

sum(df_exam['math'])/20
sum(df_exam['english'])/20
sum(df_exam['science'])/20

df_exam.shape
len(df_exam)
df_exam.size


?pd.read_excel()
import numpy as np
df_exam['total'] = df_exam['math']+df_exam['english']+df_exam['science']
df_exam
df_exam['mean'] = df_exam['total']/3
df_exam

df_exam[df_exam['math'] > 50]
df_exam[(df_exam['math'] > 50) & (df_exam['english'] > 50)]

# 예제1 :  수학 평균 이상이면서 영어 평균 아래인 사람
mean_m = np.mean(df_exam['math'])
mean_e = np.mean(df_exam['english'])
df_exam[(df_exam['math'] > mean_m) & (df_exam['english'] < mean_e)]

# 3반만
df_exam[df_exam['nclass']==3][['math', 'english', 'science']]
# 다른방법(변수사용)
df_nc3 = df_exam[df_exam['nclass']==3]
df_nc3[['math', 'english', 'science']]


a = np.array([4,2,5,3,6])
a[3]
df_exam[0:10]
df_exam[0:10:2]

df_exam.sort_values("math", ascending=False) # 내림차순
df_exam.sort_values(["nclass","math"], ascending=[True, False])


a
np.where(a > 3) #조건에 만족하는 위치를 "튜플"로 반환
np.where(a > 3, "Up", "Down") #"numpy array" 형태로 반환

df_exam["updown"] = np.where(df_exam['math'] > 50, "Up", "Down")
df_exam
