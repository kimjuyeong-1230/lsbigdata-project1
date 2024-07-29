import numpy as np
import pandas as pd

mpg = pd.read_csv('data/mpg.csv')
mpg.shape


!pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt

# 산점도 그리기
sns.scatterplot(data= mpg, x = "displ", y="hwy", hue="drv") \
              .set(xlim=[3,6], ylim=[10,30])
plt.show()

plt.clf()


# 막대그래프
df_mpg = mpg.groupby("drv", as_index=False) \
   .agg(mean_hwy = ("hwy", "mean"))
df_mpg

sns.barplot(data = df_mpg,
            x = "drv", y="mean_hwy", hue = "drv")
plt.show()


sns.barplot(data = df_mpg.sort_values("mean_hwy"),
            x = "drv", y="mean_hwy", hue = "drv")
plt.show()


# 빈도 막대 그래프 만들기
#df_mpg = mpg.groupby('drv', as_index = False) \
            .agg(n = ('drv', 'count'))
#df_mpg

#sns.barplot(data = df_mpg, x = 'drv', y = 'n')
#plt.show()
#plt.clf()
# barplot이랑 seaborn 차이
# barplot: 필요한 데이터만?(ex.3행 2열), seaborn: 원데이터 다 들어감 



#### 240729 ####
## p.212
import pandas as pd

economics = pd.read_csv('data/economics.csv')
economics.head()
economics.info()

sns.lineplot(data = economics, x='date', y='unemploy')
plt.show()
plt.clf()

economics['date2'] = pd.to_datetime(economics['date'])
economics.info()

economics[['date', 'date2']]
economics['date2'].dt.year
economics['date2'].dt.month
economics['date2'].dt.day
economics['date2'].dt.month_name()
economics['date2'].dt.quarter
economics['quarter'] = economics['date2'].dt.quarter

economics[['date', 'quarter']]

# 각 날짜는 무슨 요일인가?
economics['date2'].dt.day_name()

economics['date2'] + pd.DateOffset(days=30)
economics['date2'] + pd.DateOffset(months=1)

# 연도변수 추가 후 그래프만들기
economics['year'] = economics['date2'].dt.year
economics.head()
sns.lineplot(data = economics, x='year', y='unemploy')
plt.show()
plt.clf()

# 신뢰구간 제거(errorbar=None)
sns.lineplot(data=economics, x='year', y='unemploy', errorbar=None)
plt.show()
plt.clf()

sns.scatterplot(data=economics, x='year', y='unemploy', size=1)
plt.show()
plt.clf()


economics.head()
my_df = economics.groupby("year", as_index=False) \
         .agg(mon_mean = ("unemploy", "mean"),
         mon_std=("unemploy", "std"),
         mon_n=("unemploy", "count"))
my_df
mean + 1.96*std/sqrt(12)
my_df["left_ci"] = my_df["mon_mean"] - 1.96*my_df["mon_std"]/np.sqrt(my_df["mon_n"])
my_df["right_ci"] = my_df["mon_mean"] + 1.96*my_df["mon_std"]/np.sqrt(my_df["mon_n"])
my_df.head()


import matplotlib.ptplot as plt

x = my_df["year"]
y = my_df["mon_mean"]
plt.plot(x,y,color="black")
plt.scatter(x, my_df["left_ci"])
plt.scatter(x, my_df["light_ci"])
plt.show()
plt.clf()





