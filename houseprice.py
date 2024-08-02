#### 240724 ####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
house_df = pd.read_csv("./data/houseprice/train.csv")
house_df.shape

price_mean = house_df["SalePrice"].mean()
price_mean


sub_df = pd.read_csv("./data/houseprice/sample_submission.csv")
sub_df

sub_df["SalePrice"] = price_mean
sub_df

sub_df.to_csv("sample_submission.csv", index = False) 
sub_df



#### 240729 ####
house_df = pd.read_csv("./data/houseprice/train.csv")
house_df.columns

df_year = house_df.groupby("YearBuilt", as_index=False) \ 
                  .agg(y=("SalePrice", "mean"))
df_year

test = pd.read_csv("./data/houseprice/test.csv")
house_test = pd.merge(test, df_year, how='left',on='YearBuilt')
house_test

sub_df = pd.read_csv("./data/houseprice/sample_submission.csv")
sub_df['SalePrice'] = df['y']
sub_df

sub_df.to_csv("sample_submission.csv", index = False) 
sub_df

#"SalePrice" 열의 결측값을 해당 열의 평균값(mean)으로 대체
mean = sub_df["SalePrice"].mean()
sub_df["SalePrice"].fillna(mean)
sub_df
sub_df.to_csv("sample_submission.csv", index = False) 
sub_df


#### 선생님이 한 방법
house_train = pd.read_csv("./data/houseprice/train.csv")
house_train.columns

house_mean = house_train.groupby("YearBuilt", as_index=False) \ 
                  .agg(mean_year=("SalePrice", "mean"))
house_mean

house_test = pd.read_csv("./data/houseprice/test.csv")
house_test = pd.merge(test, df_year, how='left',on='YearBuilt')
house_test
house_test=house_test.rename(column={'mean_year':'SalePrice'})
house_test

house_test["SalePrice"].insa().sum()

# 비어있는 테스트세트 집들 확인
house_test.loc[house_test["SalePrice"].insa()]

# 평균값으로 채우기(결측치 처리)
house_mean = house_train["SalePrice"].mean()
house_test["SalePrice"] = house_test["SalePrice"].fillna(house_mean)

# sub 데이터 불러오기
sub_df = pd.read_csv("./data/houseprice/sample_submission.csv")
# SalePrice 바꿔치기
sub_df['SalePrice'] = house_test["SalePrice"]

sub_df.to_csv("sample_submission2.csv", index = False)


####### proj ####### 
#### 변수 여러개 groupby ####
house_train = pd.read_csv("./data/houseprice/train.csv")
house_test = pd.read_csv("./data/houseprice/test.csv")

df = house_train.groupby(["YearBuilt", "Neighborhood"], as_index=False) \ 
                  .agg(y=("SalePrice", "mean"))
df

house_test = pd.merge(house_test, df, how='left',on=['YearBuilt', 'Neighborhood'])
house_test

sub_df = pd.read_csv("./data/houseprice/sample_submission.csv")
sub_df['SalePrice'] = house_test['y']
sub_df

mean = sub_df["SalePrice"].mean()
sub_df["SalePrice"].fillna(mean, inplace=True)
sub_df
sub_df.to_csv("sample_submission3.csv", index = False) 
sub_df

### 결과 : 0.29


# 새로운거 도전
house_train = pd.read_csv("./data/houseprice/train.csv")
house_test = pd.read_csv("./data/houseprice/test.csv")

df = house_train.groupby(["Neighborhood"], as_index=False) \ 
                  .agg(y=("SalePrice", "mean"))
df

house_test = pd.merge(house_test, df, how='left',on='Neighborhood')
house_test

sub_df = pd.read_csv("./data/houseprice/sample_submission.csv")
sub_df['SalePrice'] = house_test['y']
sub_df

mean = sub_df["SalePrice"].mean()
sub_df["SalePrice"].fillna(mean, inplace=True)
sub_df
sub_df.to_csv("sample_submission4.csv", index = False) 
sub_df

### 결과: 0.26



# 새로운거 도전2
house_train = pd.read_csv("./data/houseprice/train.csv")
house_test = pd.read_csv("./data/houseprice/test.csv")

df = house_train.groupby(["OverallCond"], as_index=False) \ 
                  .agg(y=("SalePrice", "mean"))
df

house_test = pd.merge(house_test, df, how='left',on='OverallCond')
house_test

sub_df = pd.read_csv("./data/houseprice/sample_submission.csv")
sub_df['SalePrice'] = house_test['y']
sub_df

mean = sub_df["SalePrice"].mean()
sub_df["SalePrice"].fillna(mean, inplace=True)
sub_df
sub_df.to_csv("sample_submission5.csv", index = False) 
sub_df



#### 240731 ####
house = pd.read_csv("./data/houseprice/train.csv")
house.shape # (1460, 81)
house.tail(10).info()

# GarageYrBlt: 차고 건축 연도
# GarageQual: 차고 품질

house["GarageYrBlt"].min() #1900
house["GarageYrBlt"].max() #2010

# 1960년대 이후만 확인
house = house[house["GarageYrBlt"] >= 1960]
house.shape # (1064, 81)

df = house[["GarageYrBlt", "GarageQual"]]
df


# 10년 단위로 구간 나누기
vec_x=np.array([1960, 1970, 1980, 1990, 2000, 2010])
labels = ["1960s", "1970s", "1980s", "1990s", "2000s"]
#pd.cut(vec_x, bins = labels, right=False)

df['year_group'] = pd.cut(df['GarageYrBlt'],
                    bins=vec_x,
                    labels=labels,
                    right=False)  
df

# 연도별 GarageQual 빈도 계산
grouped = df.groupby('year_group')['GarageQual'].value_counts().unstack(fill_value=0)
grouped

import matplotlib.pyplot as plt

# 막대 그래프 시각화
grouped.plot(kind='bar', figsize=(4, 3))
plt.title('GarageQual Counts by Year Group')
plt.xlabel('Year Group')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='GarageQual', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
plt.clf()
