import pandas as pd
#from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#1
url = "https://raw.githubusercontent.com/grbruns/cst383/master/College.csv"

#2
df = pd.read_csv(url, index_col=0)

#3
df['perc.accept'] = df['Accept'] / df['Apps'] * 100
df['perc.enroll'] = df['Enroll'] / df['Accept'] * 100

plt.figure(figsize=(10, 6))
sns.scatterplot(x='F.Undergrad', y='Expend', data=df)
plt.title('Expenditure vs. Number of Full-time Undergraduates')
plt.xlabel('Number of Full-time Undergraduates')
plt.ylabel('Expenditure per Student')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='F.Undergrad', y='Outstate', data=df)
plt.title('Out-of-state Tuition vs. Number of Full-time Undergraduates')
plt.xlabel('Number of Full-time Undergraduates')
plt.ylabel('Out-of-state Tuition')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='perc.accept', y='Outstate', data=df)
plt.title('Out-of-state Tuition vs. Acceptance Rate')
plt.xlabel('Acceptance Rate (%)')
plt.ylabel('Out-of-state Tuition')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='perc.enroll', y='Outstate', data=df)
plt.title('Out-of-state Tuition vs. Enrollment Rate')
plt.xlabel('Enrollment Rate (%)')
plt.ylabel('Out-of-state Tuition')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Grad.Rate', y='Outstate', data=df)
plt.title('Out-of-state Tuition vs. Graduation Rate')
plt.xlabel('Graduation Rate (%)')
plt.ylabel('Out-of-state Tuition')

#4
max_books_row = df[df['Books'] == df['Books'].max()]
top_grad_rate_with_max_books = max_books_row['Grad.Rate'].values[0]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Books', y='Grad.Rate', data=df)

plt.scatter(max_books_row['Books'], max_books_row['Grad.Rate'], color='red', s=100, label=f'Max Books: {df["Books"].max()}')

for index, row in max_books_row.iterrows():
    plt.text(row['Books'], row['Grad.Rate'], f' {row["Grad.Rate"]}', color='red', ha='left', va='center')

plt.title('Graduation Rate vs. Books Cost')
plt.xlabel('Books Cost')
plt.ylabel('Graduation Rate')
