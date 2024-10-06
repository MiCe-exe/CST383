import pandas as pd
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#1
wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")

#2
df = wine.sample(n=20, random_state=42)
df.reset_index(drop=True, inplace=True)

#3
scaled_df = df.apply(zscore)

#4
scaled_df.describe().loc['min']

#5
scaled_df.describe().loc['max']

#6
def unit_scaling(column):
    return (column - column.min()) / (column.max() - column.min())

unit_scaled_df = df.apply(unit_scaling)

#7
#little shocked only gives 0's or 1's
unit_scaled_df.describe().loc['min']
unit_scaled_df.describe().loc['max']

#8
df = wine
df.corr()

#9
def most_corr_index(x):
    return x.sort_values(ascending=False).index[1]
df.corr().apply(most_corr_index)

#10
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')

#11
#quality is most positively correlated with alcohol

#12
#Looks good

#13
n=10   # number of students
sem_years = np.array([s+' '+str(y) for y in np.arange(14,19) for s in ['spring', 'fall']])
gpa = np.random.normal(loc=3, scale=0.5, size=(n,sem_years.size))
gpa = np.clip(gpa, 0, 4)
gpa = pd.DataFrame(gpa, columns=sem_years)
otter_ids = pd.DataFrame({'otter_id': np.random.randint(1000, 10000, n)})
gpa_by_semester = pd.concat([otter_ids, gpa], axis=1)

#14
gpa_long = gpa_by_semester.melt(id_vars='otter_id', var_name='semester', value_name='gpa')
