# You have a CSV file named "sales_data.csv" containing sales data with columns for "Date", "Product",
# "Quantity" and "Revenue" Load this data using Pandas and answer the following questions:
# 1. How many rows and columns are there in the dataset?
# 2. What is the total revenue for all the sales?


import pandas as pd
df = pd.read_csv("sales_data.csv")
print(df.head())
print("Number of rows: ", len(df))
print("Number of columns: ", len(df.columns))
print("Total Revenue: ", sum(df['Revenue']))





# You have a DataFrame called "student_data" with columns "Student_ID," "Name," "Age," and "GPA."
# Perform the following operations using Pandas:
# 1. Filter and display the rows of students who are 20 years old or older.
# 2. Calculate the average GPA of the students in the DataFrame.
# 3. Sort the DataFrame in descending order of GPA and display the top 5 students with the highest
# GPAs.
# 4. Group the students by their ages and calculate the average GPA for each age group.




import pandas as pd
df = pd.read_csv("student_data.csv")
print(df.head(), "\n")
age = df[df['Age'] > 19]
print("Students who are 20 years old or older: \n", age)
print("\nAverage GPA of all students:",df['GPA'].mean().round(2))
data1 = df.sort_values(by='GPA', ascending= False)
print("\nTop 5 students with highest GPA: \n", data1.head(5))
data2 = df.groupby('Age')['GPA'].mean().reset_index()
print("\nAverage GPA by age group: \n", data2)