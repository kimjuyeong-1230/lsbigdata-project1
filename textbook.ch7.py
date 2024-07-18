import pandas as pd
import numpy as np

df = pd.DataFrame({"sex" : ['M', 'F', np.nan, 'M', 'F'],
                  "score" : [5,4,3,4,np.nan]})
df

df["score"]+1


# 데이터프레임 location을 사용한 인덱싱
exam = pd.read_csv("data/exam.csv")
exam.loc[[2,7,14],]  # loc: 라벨이름
exam.iloc[0,0] #iloc: 정수 인덱스 

exam.loc[[2,7,4], ["math"]] = np.nan
exam.iloc[[2,7,4],2] = np.nan


df[df["score"]==3.0, ["score"]]=4 #시리즈라 오류

df.loc[df["score"]==3.0, "score"]=4
df


# 수학점수 50점 이하인 학생들 점수 50점으로 상향조정
exam
exam.loc[exam["math"] <= 50 , "math"] = 50

# 영어점수 90점 이상 90점으로 하향 조정
exam.iloc[exam["english"] >= 90, 3]=90
exam.iloc[exam[exam["english"] >= 90].index, 3]=90
exam.iloc[np.where(exam["english"] >= 90)[0], 3]=90

# math 점수 50점 이하 "-" 변경
exam.loc[exam["math"] <= 50, "math"] = "-"
exam

# "-" 결측치를 수학점수 평균 바꾸고 싶은 경우
# 방법 세가지
math_mean = exam[exam["math"] != "-"]["math"].mean() #방법1
math_mean = exam.query('math not in ["-"]').mean() #방법2
math_mean = exam.loc[(exam["math"] != "-"), "math"].mean() #방법3

exam.loc[exam["math"] == "-", "math"] = math_mean
exam

# 방법4
exam.loc[exam["math"]=="-", ["math"]]=np.nan
math_mean = exam["math"].mean()  # nan값은 무시하고 계산
exam.loc[pd.isna(exam["math"]), ["math"]]=math_mean
exam

#방법5
math_mean = exam.loc[(exam["math"]=="-", math_mean, exam["math"])]
exam["math"] = np.where(exam["math"] == "-", math_mean, exam["math"])
exam

#방법6
math_mean = np.nonmean(np.array([np.nan if x == "-" else float(x) for x in exam["math"]]))
np.array([float(x) if x != "-" else np.nan for x in exam["math"]])
exam["math"] = np.where(exam["math"] == "-", math_mean, exam["math"])
exam

#방법7
math_mean = exam[exam["math"] != "-"]["math"].mean()
exam["math"] = exam["math"].replace("-", math_mean) #replace: 값 치환 ("-"를 math_mean으로)



# 결측치 확인
pd.isna(df)  # nan값만 true값
pd.isna(df).sum() # nan값 카운트

# 결측치 제거
df.dropna(subset="score")  #subset: 특정 부분 선택
df.dropna(subset=["score", "sex"])

# 결측치 대체


