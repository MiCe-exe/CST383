import pandas as pd
from matplotlib import pyplot as plt

infile = "https://raw.githubusercontent.com/grbruns/cst383/master/campaign-ca-2016-sample.csv"
df = pd.read_csv(infile)

#2
df.info()

#3
#I dont see any that should be numbers

#4
df.isna().sum()[df.isna().sum() > 0]
df.isna().sum().sum()

#5
#There is some NAN in other fields. The "contbr_employer" has NAN as values. 
#Also "contbr_occupation" has NAN values.
df.contbr_occupation.value_counts().head(n=20)

#6
#Yes
df['contbr_employer'].isna().sum()

#7
#AI Overview
#Learn more MR can stand for Medical Representative I would say no
#Yes I see repeated values formatted the same way. RN and R.N.
df.contbr_occupation.value_counts().sort_values(ascending=True).head(n=100)
df['contbr_occupation'].value_counts()[df['contbr_occupation'].value_counts() > 4].sort_values(ascending=True).head(20)

#8
#I had to look on stack overflow for this not sure if this is the most correct way of doing this.
(df['memo_cd']
 .pipe(lambda x: pd.Series({
     'num_unique_values': len(x.unique()),
     'unique_values': x.value_counts(dropna=False),
     'fraction_na': x.isnull().mean()
 }))
)


#9
#
df['contb_receipt_amt'].plot.hist(bins=30, edgecolor='black')
plt.title('Histogram of Contribution Amounts')
plt.xlabel('Contribution Amount')
plt.ylabel('Frequency')

#10
df['contb_receipt_amt'].describe()[['min', 'max']]
df[df['contb_receipt_amt'] < 0]['contb_receipt_amt'].value_counts().head()

#11
(df['contbr_zip']
 .str.len()
 .pipe(lambda x: pd.Series({'all_same_length': x.nunique() == 1, 'zip_code_length_counts': x.value_counts()}))
)
#we can still use the zip code if we ignroe the last 4 numeric numbers.
zip_code = df['contbr_zip'] = df['contbr_zip'].str[:5]

#12
employer_lengths = df['contbr_employer'].str.len()

# Create a histogram of the lengths
plt.hist(employer_lengths, bins=20, edgecolor='black')

# Set title and labels
plt.title('Histogram of Lengths of contbr_employer Values')
plt.xlabel('Length of contbr_employer')
plt.ylabel('Frequency')

#13
min_amt = df['contb_receipt_amt'].min()
max_amt = df['contb_receipt_amt'].max()
df['s_amt1'] = (df['contb_receipt_amt'] - min_amt) / (max_amt - min_amt)
df.s_amt1.head()
