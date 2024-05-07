#Read data from json or csv file and convert to data frame
#install.packages("jsonlite")
library(jsonlite)

# Read the JSON file into R
json_data <- fromJSON("E:/SEM 6/DATA SCIENCE/practice/1-jprac.json")

# Read the CSV file into R
csv_data <- read.csv("E:/SEM 6/DATA SCIENCE/practice/1-prac.csv")

# Convert JSON data to dataframe
df <- as.data.frame(json_data)
# Convert CSV data to dataframe
df2 <- as.data.frame(csv_data)

# perform basic pre-processing task
View(df)
# View the first few rows of the dataframe
head(df)
summary(df)
mean(df$Age)
median(df$Age)
sum(df$Age)
unique(df$City)
# Remove duplicate rows
df <- df[!duplicated(df), ]
df

# Convert Age column to integer (if needed)
df$Age <- as.integer(df$Age)
df



# Filtering , sorting , grouping

# filter age > 30 from above data set
filtered_df <- subset(df, City == 'New York')
filtered_df
filtered_df <- subset(df, Age == 30)
filtered_df

# sorting
# Sort the dataframe based on a specific column (e.g., 'Age')
sorted_df <- df[order(df$Age), ]
sorted_df

#grouping
# Group the dataframe based on 'Age' and aggregate the 'Name' column
grouped_df <- aggregate(Name ~ Age, data = df, FUN = function(x) paste(x, collapse = ", "))
grouped_df
