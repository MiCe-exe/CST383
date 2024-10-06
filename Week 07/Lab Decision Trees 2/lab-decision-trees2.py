import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#1 about 0.47 or 0.46875

#2
def gini(class_counts):
    if sum(class_counts) == 0:
        return 0
    p = class_counts[0]/sum(class_counts)
    return 2 * p * (1 - p)

#3
gini([30,50])
gini([10,10])
gini([20,0])
gini([100,0])

#4
df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/heart.csv")
df['output'] = df['output'] - 1

df = df[['age', 'maxhr', 'restbp', 'output']]
sns.scatterplot(x='age', y='maxhr', hue='output', data=df)

#5

#6
gini_root = gini([(df['output'] == i).sum() for i in [0,1]])


#7
split_val = 50
df_lo = df[df['age'] < split_val]
df_hi = df[df['age'] >= split_val]

counts_lo = [(df_lo['output'] == i).sum() for i in [0,1]]
counts_hi = [(df_hi['output'] == i).sum() for i in [0,1]]

gini_lo = gini(counts_lo)
gini_hi = gini(counts_hi)

#8
fraction_lo = df_lo.shape[0]/df.shape[0]
fraction_hi = df_hi.shape[0]/df.shape[0]
gini_split = fraction_lo * gini_lo + fraction_hi * gini_hi

#9
# Split with 50 is better than the split with 40
count_age_lt_40 = df[df['age'] < 40]['output'].value_counts()

p0_40 = count_age_lt_40.get(0, 0) / count_age_lt_40.sum()
p1_40 = count_age_lt_40.get(1, 0) / count_age_lt_40.sum()
gini_lt_40 = 2 * p0_40 * p1_40

count_age_lt_50 = df[df['age'] < 50]['output'].value_counts()

p0_50 = count_age_lt_50.get(0, 0) / count_age_lt_50.sum()
p1_50 = count_age_lt_50.get(1, 0) / count_age_lt_50.sum()
gini_lt_50 = 2 * p0_50 * p1_50

print("Gini index for age < 40:", gini_lt_40)
print("Gini index for age < 50:", gini_lt_50)


#10
age_thresholds = []
gini_values = []

for age_threshold in range(20, 81):
    count_age_lt_threshold = df[df['age'] < age_threshold]['output'].value_counts()
    
    ##count_age_lt_threshold = count_age_lt_threshold.dropna()
    
    if len(count_age_lt_threshold) > 1:
        p_0_lt_threshold = count_age_lt_threshold.get(0, 0) / count_age_lt_threshold.sum()
        p_1_lt_threshold = count_age_lt_threshold.get(1, 0) / count_age_lt_threshold.sum()
        gini_lt_threshold = 2 * p_0_lt_threshold * p_1_lt_threshold
    else:
        gini_lt_threshold = None
    
    age_thresholds.append(age_threshold)
    gini_values.append(gini_lt_threshold)

age_thresholds = [age_thresholds[i] for i in range(len(age_thresholds)) if gini_values[i] is not None]
gini_values = [gini_values[i] for i in range(len(gini_values)) if gini_values[i] is not None]

gini_df = pd.DataFrame({'Age': age_thresholds, 'Gini Value': gini_values})

##gini_df.dropna(inplace=True)

sns.scatterplot(x='Age', y='Gini Value', data=gini_df)
plt.xlabel('Age')
plt.ylabel('Gini Value')
plt.title('Gini Split Values for Different Age Thresholds (NaN values dropped)')
plt.show()

best_age_index = gini_df['Gini Value'].idxmin()
best_age_threshold = gini_df.loc[best_age_index, 'Age']
best_gini_value = gini_df.loc[best_age_index, 'Gini Value']

print("Best age value for a split on age:", best_age_threshold)
print("Corresponding Gini value:", best_gini_value)