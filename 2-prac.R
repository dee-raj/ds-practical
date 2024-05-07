# Load the required libraries
#install.packages('fastDummies')
#install.packages('scales')

library(fastDummies)
library(jsonlite)
library(scales)

# Load the JSON data
json_data <- fromJSON("E:/SEM 6/DATA SCIENCE/practice/2-data.json")
df <- as.data.frame(json_data)

# Specify the columns to standardize and normalize
columns_to_normalize <- c("Sales", "Cost", "Profit")

# Standardization
df[columns_to_normalize] <- scale(df[columns_to_normalize])


# Normalization using the scales package
df[columns_to_normalize] <- lapply(df[columns_to_normalize], rescale, to = c(0, 1))

# Print the standardized and normalized dataframe
print(df)

# dummify
cat_cols <- c("Product", "Category")
dummified_df <- dummy_cols(df, select_columns = cat_cols, remove_first_dummy = TRUE)
dummified_df
