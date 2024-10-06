import pandas as pd
import numpy as np

url = "https://raw.githubusercontent.com/grbruns/cst383/master/airquality.csv"
df = pd.read_csv(url)

#2
df.head()

#3
df.info()

#4
df.isna().sum().sum()

#5 displays 42 and 2
df.isna().any(axis=1).sum()
np.sum(pd.Series([0,1,0,3]) > 0)

#6
pd.Series([0,1,0,3]) / 10
# 0    0.0
# 1    0.1
# 2    0.0
# 3    0.3
# dtype: float64

#7
df.isna().sum(axis=1) / len(df.columns)

#8
import matplotlib.pyplot as plt
calculate_na = df.isna().mean()
calculate_na.plot(kind='bar', figsize=(10,5), color='red')

plt.title("Fraction Values")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)

#9 Depending where the data is consentrated. 

#10
df_cleanrows = df.dropna()
df_cleanrows.isna().any().any()

#11
df_cleancols = df.dropna(axis=1)

#12
df_cleancols.count().sum()
df_cleanrows.count().sum()
# df_cleancols.count().sum()
# Out[32]: 612

# df_cleanrows.count().sum()
# Out[33]: 666

#13
df_med = df.fillna(df.median())

#14
df_mean = df.fillna(df.mean())
