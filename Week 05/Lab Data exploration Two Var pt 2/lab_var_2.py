import numpy as mp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#1 & 2
df = pd.read_csv('https://raw.githubusercontent.com/grbruns/cst383/master/College.csv', index_col=0)
df.head()



#3
breaks = df['F.Undergrad'].quantile([0,0.33, 0.66, 1.0])
df['Size'] = pd.cut(df['F.Undergrad'],
   			include_lowest=True, bins=breaks,
   			labels=['small', 'medium', 'large'])

df.Size.value_counts().plot.bar()

#4
g = sns.FacetGrid(df, col='Size', height=4, aspect=0.8)
g.map(plt.scatter, 'PhD', 'Outstate', s=20, color="r")

#5
sns.scatterplot(x='PhD', y='Outstate', hue='Size', data=df, s=50)

#6
sns.scatterplot(x='PhD', y='Outstate', hue='Size', style='Size', data=df, s=55)

#7
sns.catplot(y='Outstate', col='Size', data=df, kind='violin', height=4, aspect=0.7)

#8
sns.catplot(y='Outstate', col='Size', data=df, kind='violin', inner='stick', height=4, aspect=0.7)

#9
g = sns.FacetGrid(df, row='Size', height=2.5, aspect=1.8)
g.map(plt.hist, 'PhD', color="r")

#10
df['Accept.Rate'] = df['Accept'] / df['Apps']
low_grad = df[(df['Accept.Rate'] < 0.3) & (df['Grad.Rate'] < 60)]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=low_grad, x='Accept.Rate', y='Grad.Rate', hue='Size', s=100)
plt.title('Selective Schools with Low Graduation Rates')
plt.xlabel('Acceptance Rate')
plt.ylabel('Graduation Rate')
plt.axhline(60, color='red', linestyle='--', label='Graduation Rate Threshold (60%)')
plt.axvline(0.3, color='blue', linestyle='--', label='Selectivity Threshold (30%)')
plt.legend()