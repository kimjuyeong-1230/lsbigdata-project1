#### 240716 ####
import panddas as pd

# 데이터전처리 함수
# query()
# df[]
# sort_values()
# groupby
# assign
# agg()
# merge()
# concat()


exam = pd.read_csv("data/exam.csv")

# query: 조건에 맞는 행 추출 
exam.query("nclass == 1") # exam[exam["nclass"]==1]
exam.query("nclass != 1")

exam.query("math > 50")
exam.query("math < 50")
exam.query("english >= 50")
exam.query("english <= 80")

exam.query("nclass == 1 & math >= 50")
exam.query("nclass == 2 & english >= 80")
exam.query("math >= 90 | english >= 90")
exam.query("english < 90 | science < 50")
exam.query("nclass == 1 | nclass == 3 | nclass == 5")
exam.query("nclass in [1,3,5]")
exam.query("nclass not in [1,2]")

exam[~exam["nclass"].inis([1,2])]



exam[["nclass"]]
exam.drop(columns = ["math", "english"])
exam


exam.query("nclass ==1")[["math", "english"]]
exam.query("nclass ==1") \
  [["math", "english"]] \
  .head()
  

# 정렬
exam.sort_values("math")
exam.sort_values("math", ascending = False)  #내림차순
exam.sort_values(["nclass", "english"],ascending = [True, False])


# 변수추가
exam = exam.assign(
  total = exam["math"] + exam["english"] + exam["science"],
  mean =(exam["math"] + exam["english"] + exam["science"])/3) \
  .sort_values("total", ascending=False)
exam.head()


# lambda를 이용해 데이터프레임명 줄여쓰기
exam2 = pd.read_csv("data/exam.csv")
exam2 = exam.assign(
  total = lambda x : x["math"] + x["english"] + x["science"],
  mean =lambda x : x["total"]/3) \
  .sort_values("total", ascending=False)
exam2.head()


# 그룹나눠 요약하는 groupby() + agg()
exam2.agg(mean_math = ("math", "mean"))
exam2.groupby("nclass") \
    .agg(mean_math = ("math", "mean"))


exam2.groupby("nclass") \
    .agg(mean_math = ("math", "mean"),
         )
         
import pydataset

df = pd.read_csv("data/mpg.csv")
df.query('category == "suv"') \
  .assign(total = (df['hwy'] + df['cty']) / 2) \
  .groupby('manufacturer') \
  .agg(mean_tot = ('total', 'mean')) \
  .sort_values('mean_tot', ascending = False) \
  .head()
