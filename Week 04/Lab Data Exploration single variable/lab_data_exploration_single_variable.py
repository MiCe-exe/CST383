import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#1
url = "https://raw.githubusercontent.com/grbruns/cst383/master/1994-census-summary.csv"

#2
df = pd.read_csv(url)

#4
df.head()
df.tail()
df.info()
df.describe()


#5
df = df.drop(['usid', 'fnlwgt'], axis=1)

#6
df.head()

#7
edu_num_desc = df['education_num'].describe()

min_edu_num = edu_num_desc['min']
max_edu_num = edu_num_desc['max']
median_edu_num = edu_num_desc['50%']

plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='education_num', bins=20, kde=True)
plt.xlabel('Years of Education')
plt.ylabel('Frequency')
plt.title('Histogram of Years of Education')
plt.grid(True)

plt.axvline(x=min_edu_num, color='r', linestyle='--', label='Min Education Years')
plt.axvline(x=max_edu_num, color='g', linestyle='--', label='Max Education Years')
plt.axvline(x=median_edu_num, color='b', linestyle='--', label='Median Education Years')
plt.legend()

#8
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='education_num', bins=20, kde=True)
plt.xlabel('Years of Education')
plt.ylabel('Frequency')
plt.title('Histogram of Years of Education')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='education_num')
plt.xlabel('Years of Education')
plt.ylabel('Count')
plt.title('Count of Rows by Years of Education')
plt.grid(True)

#9
plt.figure(figsize=(8, 6))
sns_plot = sns.kdeplot(data=df, x='capital_gain', fill=True)
plt.xlabel('Capital Gain')
plt.ylabel('Density')
plt.title('Density Plot of Capital Gain')
plt.grid(True)

#sns_plot.figure.savefig('output.png')
sns_plot.get_figure().savefig('output.png')

#10
plt.figure(figsize=(10, 6))
sns_plot = sns.countplot(data=df, x='workclass', order=df['workclass'].value_counts().index)
plt.xlabel('Workclass')
plt.ylabel('Count')
plt.title('Count of Rows by Workclass')
plt.xticks(rotation=45)
plt.grid(True)


#11
sex_counts = df['sex'].value_counts()
sex_genders = sex_counts / len(df)

plt.figure(figsize=(8, 6))
sns_plot = sns.barplot(x=sex_genders.index, y=sex_genders.values)
plt.xlabel('Sex')
plt.ylabel('Fraction of Rows')
plt.title('Distribution of Sex')

for i, fraction in enumerate(sex_genders):
    plt.text(i, fraction + 0.01, f'{fraction:.2%}', ha='center')
    
#12
marital_status = df['marital_status'].value_counts()

plt.figure(figsize=(10, 6))
sns_plot = sns.barplot(x=marital_status.values, y=marital_status.index, orient='h')
plt.xlabel('Count')
plt.ylabel('Marital Status')
plt.title('Distribution of Marital Status')
plt.grid(True)