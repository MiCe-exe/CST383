import numpy as np
import pandas as pd
from matplotlib import rcParams
import seaborn as sns
from scipy.stats import zscore

df = pd.read_csv('https://raw.githubusercontent.com/grbruns/cst383/master/iris.csv',index_col=0)

species = df.drop(columns='species').corr()

species.info()

# Count the number of samples for each species
species_counts = df['species'].value_counts()

# Calculate the total number of samples
total_samples = len(df)

# Calculate the marginal probability of 'setosa'
marginal_probability_setosa = species_counts['setosa'] / total_samples

# Print the marginal probability
print(f"The marginal probability that an iris is of species 'setosa' is {marginal_probability_setosa:.4f}")
print(marginal_probability_setosa)




# Load the dataset
import pandas as pd

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/grbruns/cst383/master/iris.csv', index_col=0)
df.info()
# Filter the dataset for samples with sepal length less than 5

# Filter the dataset for samples with sepal length less than 5
filtered_df = df[df.index < 5]

# Count the total number of samples in the filtered dataset
total_filtered_samples = len(filtered_df)

# Count the number of 'setosa' samples in the filtered dataset
setosa_filtered_samples = len(filtered_df[filtered_df['species'] == 'setosa'])

# Calculate the conditional probability
conditional_probability_setosa = setosa_filtered_samples / total_filtered_samples

# Print the conditional probability
print(f"The conditional probability that an iris is of species 'setosa' given its sepal length is less than 5 is {conditional_probability_setosa:.4f}")




# Filter the dataset for samples with species 'setosa'
setosa_df = df[df['species'] == 'setosa']

# Count the total number of 'setosa' samples
total_setosa_samples = len(setosa_df)

# Count the number of 'setosa' samples with sepal length less than 5
setosa_sepal_length_less_than_5 = len(setosa_df[setosa_df.index < 5])

# Calculate the conditional probability
conditional_probability_sepal_length_less_than_5 = setosa_sepal_length_less_than_5 / total_setosa_samples

# Print the conditional probability
print(f"The conditional probability that an iris has sepal length less than 5 given its species is 'setosa' is {conditional_probability_sepal_length_less_than_5:.4f}")





# Filter the dataset for samples with species 'versicolor'
versicolor_df = df[df['species'] == 'versicolor']

# Count the total number of 'versicolor' samples
total_versicolor_samples = len(versicolor_df)

# Initialize conditional probability
conditional_probability = 0.0

# Check if there are 'versicolor' samples
if total_versicolor_samples > 0:
    # Count the number of 'versicolor' samples with sepal length greater than 6 and sepal width less than 2.6
    versicolor_condition_samples = len(versicolor_df[(versicolor_df.index > 6) & (versicolor_df['sepal_width'] < 2.6)])

    # Calculate the conditional probability
    conditional_probability = versicolor_condition_samples / total_versicolor_samples

# Print the conditional probability
print(f"The conditional probability that an iris has sepal length greater than 6 and sepal width less than 2.6 given its species is 'versicolor' is {conditional_probability:.4f}")




# Compute the correlation matrix, omitting the 'species' column
correlation_matrix = df.drop(columns='species').corr()

# Find the pairs of columns with the highest and lowest correlation values
max_correlation = correlation_matrix.unstack().nlargest(4)
min_correlation = correlation_matrix.unstack().nsmallest(4)

# Print the most strongly correlated columns
print("Most strongly positively correlated columns:")
print(max_correlation)
print("\nMost strongly negatively correlated columns:")
print(min_correlation)
