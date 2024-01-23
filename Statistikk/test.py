import pandas as pd
import numpy as np

df = pd.read_csv("data1.csv")
#print(df.values)
#df.to_numpy().flatten()#
mean = np.mean(df.values)
round_mean = np.round(mean, 2)
print(f"Mean: {round_mean}")

#Standard deviation
std = np.std(df.values, 0)
print(std)
round_std = np.round(std, 2)
print(f"Std deviation {round_std}")

#Std error of mean
#print(df.size)
std_err = np.std(df.values, ddof=1) / np.sqrt(df.size)
std_err = np.round(std_err, 3)
print(f"Std error: {std_err}")




# How many samples within std dev: GPT
# Example dataset
dataset = df.to_numpy().flatten()

# Calculate mean and standard deviation
mean = np.mean(dataset)
std = np.std(dataset)

# Number of standard deviations to check
num_stdev = 2

# Define range
range_min = mean - (num_stdev * std)
range_max = mean + (num_stdev * std)

# Check samples within range
in_range = np.logical_and(dataset > range_min, dataset < range_max)

# Count number of samples within range
num_samples_within_stdev = np.count_nonzero(in_range)
percent = num_samples_within_stdev / 100
print(percent*100)

print(f"Number of samples within {num_stdev} standard deviations: {num_samples_within_stdev}")