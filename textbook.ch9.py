#!pip install pyreadstat
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyreadstat

raw_welfare = pd.read_spss("./data/koweps/Koweps_hpwc14_2019_beta2.sav")
raw_welfare


welfare = raw_welfare.copy()
welfare.shape
welfare.describe()

welfare = welfare.rename(
  columns = {
             'h14_g3' : 'sex',  #성별
             'h14_g4' : 'birth',  # 태어난 연도 
             'h14_g10' : 'marriage_type',  #혼인상태
             'h14_g11' : 'religion',  # 종교
             'p1402_8aq1' : 'income',  # 월급
             'h14_eco9' : 'code_job',  # 직업코드
             'h14_reg7' : 'code_region'  #지역코드
})

welfare = welfare[['sex', 'birth', 'marriage_type', 'religion','income', 'code_job','code_region']]
welfare.shape
welfare.info()


### 성별에 따른 월급 차이
welfare.dtypes
welfare["sex"].dtypes
welfare["sex"].value_counts()
welfare["sex"].isna().sum()

welfare["sex"] = np.where(welfare["sex"] == 1, 'male', 'female')
welfare["sex"].value_counts()


welfare["income"].describe()
welfare["income"].isna().sum()

sex_income = welfare.dropna(subset = "income") \
       .groupby("sex", as_index= False) \
       .agg(mean_income = ("income", "mean"))
sex_income

sns.barplot(data = sex_income, x = "sex", y="mean_income", hue="sex")
plt.show()
plt.clf()


# 숙제: 위 그래프에서 각 성별 95% 신뢰구간 계산 후 그리기
# 위아래 검정색 막대기로 표시


### 나이와 월급의 관계
welfare["birth"].describe()
sns.histplot(data=welfare, x = "birth")
plt.show()
plt.clf()

welfare["birth"].isna().sum()

welfare = welfare.assign(age= 2019 - welfare["birth"] + 1)
sns.histplot(data = welfare, x="age")
plt.show()
plt.clf()


# 나이에 따른 월급 평균표
age_income = welfare.dropna(subset = "income") \
       .groupby("age") \
       .agg(mean_income = ("income", "mean"))
age_income.head()

sns.lineplot(data = age_income, x = "age", y="mean_income")
plt.show()
plt.clf()

# 나이별 income na개수 세기
my_df = welfare.assign(income_na = welfare["income"].isna()) \
               .groupby("age", as_index = False) \
               .agg(n = ("income_na", "count"))
my_df

sns.histplot(data = my_df, x='age')
plt.show()
plt.clf()


### 연령대별 월급 차이
welfare["age"].head()
welfare = welfare.assign(ageg = np.where(welfare["age"]<30, "young",
                                np.where(welfare["age"]<=59, "middle", "old")))

welfare

welfare["ageg"].value_counts()
sns.countplot(data = welfare, x="ageg", hue="ageg")
plt.show()
plt.clf()

# 연령대별 월급 평균표 만들기
ageg_income = welfare.dropna(subset="income") \
                     .groupby("ageg", as_index=False) \
                     .agg(mean_income = ("income", "mean"))
ageg_income

sns.barplot(data = ageg_income, x = "ageg", y="mean_income")
plt.show()
plt.clf()

sns.barplot(data = ageg_income, x = "ageg", y="mean_income",
            order = ["young", "middle", "old"])
plt.show()
plt.clf()


# 나이대별로(10대, 20대, 30대 ...)
vec_x = np.random.randint(0,100,50)
bin_cut=np.array([0,9,19,29,39,49,59,69,79,89,99,109,119])
pd.cut(vec_x, bins = bin_cut)


welfare = welfare.assign(age_group=pd.cut(welfare["age"],
                    bins=bin_cut,
                    labels=(np.arange(12)*10).astype(str)+"대"))
welfare["age_group"]


age_income = welfare.dropna(subset="income") \
                     .groupby("age_group", as_index=False) \
                     .agg(mean_income = ("income", "mean"))
age_income

sns.barplot(data = age_income, x="age_group", y="mean_income")
plt.show()
plt.clf()

## 한글
from matplotlib import font_manager, rc

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # 예시: 윈도우 시스템에 있는 맑은 고딕 폰트 경로
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)


# 변수타입이 카테고리로 설정되어 있으면, groupby+agg 안됨
# 그래서 object로 바꿔줌
welfare["age_group"] = welfare["age_group"].astype("object")

sex_age_income = welfare.dropna(subset = "income") \
    .groupby(["age_group", "sex"], as_index = False) \
    .agg(mean_income = ("income", "mean"))
sex_age_income

sns.barplot(data=sex_age_income,
            x = "age_group", y="mean_income",
            hue="sex")
plt.show()
plt.clf()

# 연령대별, 성별 상위 4% 수입 찾아보기
sex_age_income = welfare.dropna(subset = "income") \
    .groupby(["age_group", "sex"], as_index = False) \
    .agg(top4per_income = ("income", lambda x : np.quantile(x, q=0.96)))
sex_age_income


### 직업별 월급차이
welfare["code_job"].dtypes
welfare["code_job"].value_counts()

# 직종 데이터 불러오기
list_job = pd.read_excel("./data/koweps/Koweps_Codebook_2019.xlsx",
                         sheet_name = "직종코드")
list_job.head()

welfare = welfare.merge(list_job, how= 'left', on="code_job")

# 직업별 월급 차이 분석하기
df = welfare.dropna(subset= ['job', 'income']) \
                    # query("sex == 'female'") \
                    .groupby('job', as_index = False) \
                    .agg(mean_income = ("income", "mean")) \
                    .sort_values("mean_income", ascending=False) \
                    .head(10)
df                  
                  
plt.rcParams.update({'font.family':'Malgun Gothic'})
sns.barplot(df, y='job', x='mean_income', hue='job')
plt.show()
plt.clf()


### 종교 유무에 따른 이혼율 분석하기(9-8)
# 이혼율 표
welfare.info()
welfare["marriage_type"]
df = welfare.query("marriage_type!=5")\
            .groupby("religion", as_index = False)\
            ["marriage_type"] \
            .value_counts(normalize = True)  #비율구하기
df

df = df.query("marriage_type == 1")\
       .assign(proportion = df["proportion"]*100) \
       .round(1)  #소수 첫째자리에서 반올림
df

# 그래프
rel_div = rel_div.query('marriage == "divorce"') \
                 .assign(proportion = rel_div["proportion"]*100)\
                 .round(1)








