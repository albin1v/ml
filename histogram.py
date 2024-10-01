# Question: Use a dataset of your choice (e.g., exam scores of students, employee salaries, or
# any other numerical data). Create a histogram to visualize the data's distribution. Afterward,
# plot quartiles (e.g., Q1, Q2, Q3) on the same graph. Answer the following questions:
# 1. What does the histogram reveal about the data's distribution?
# 2. How do the quartiles relate to the histogram?
# 3. Are there any outliers in the data, and if so, how do they affect the quartiles?
# b. Output: Provide the histogram and quartile plot along with a written analysis.


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
marks = np.array([10,18,34,37, 33, 38, 34, 24, 80, 45, 49, 27, 31, 35, 42])
fig, ax = plt.subplots(figsize =(4, 4))
ax.hist(marks, color = "darkcyan", ec="black", lw=1)
plt.title('Histogram: Exam Score')
plt.ylabel('No. of students')
plt.xlabel('Score')
plt.figure(figsize=(4, 4))
sns.boxplot(y=marks, color='darkcyan')
plt.title('Quartile Plot: Exam Score')
plt.ylabel('Score')
plt.show()




# a. Question: Choose a dataset that contains two numerical variables (e.g., income vs. education
# level, temperature vs. ice cream sales). Create a distribution chart for each variable and a
# scatter plot to visualize their relationship. Answer the following questions:
# 1. What do the distribution charts reveal about each variable?
# 2. Is there a correlation between the two variables based on the scatter plot?
# 3. Can you identify any patterns or trends in the data?
# Output: Present the distribution charts, scatter plot, and your observations in a report



#Heat Map charts
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
x_values = [1, 2, 3, 4, 5]
y_values = [10, 15, 13, 18, 20]
data_values = [10, 15, 13, 18, 20]
df = pd.DataFrame({'x': x_values, 'y': y_values, 'value': data_values})
heatmap_data = df.pivot_table(index='x', columns='y', values='value')
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', cbar=True)
plt.title('Heatmap Example')
plt.show()
#Scatter Plot
x_values = [1, 2, 3, 4, 5]
data_values = [10, 15, 13, 18, 20]
df = pd.DataFrame({'x': x_values, 'value': data_values})
plt.scatter(df['x'], df['value'], marker='o', color='blue', label='Data Points')
plt.xlabel('No. of students')
plt.ylabel('Values')
plt.title('Scatter Plot based on Heatmap Values')
plt.show()