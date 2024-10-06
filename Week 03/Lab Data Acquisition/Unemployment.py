import pandas as pd
import seaborn as sns
from matplotlib import rcParams

# allow output to span multiple output lines in the console
pd.set_option('display.max_columns', 500)

# switch to seaborn default stylistic parameters
# see the useful https://seaborn.pydata.org/tutorial/aesthetics.html
sns.set()
sns.set_context('paper') # 'talk' for slightly larger

# change default plot size
rcParams['figure.figsize'] = 9,7

#5. Load file
df = pd.read_csv("unemployment.csv")

#Get some info of the Dataset
print(df.head())

#7. You don't want any erros with the data because it can be a difference of government funded program assitiing families 
#   that havea family member unemployed. So you have to have the data be able to replicable the data without issues. 

#8. With the excel sheet it is listed top of the row where the data coems from. You also have a second sheet with 
#   variables definitions