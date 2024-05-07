# Load the necessary library
#install.packages("ggplot2")
# Load the necessary library
library(ggplot2)

# Load the iris data set
data(iris)

# Perform the linear regression
model <- lm(Sepal.Length ~ Petal.Length, data = iris)

# Print the summary of the model
print(summary(model))

# Create a scatter plot of the data and add the regression line
ggplot(iris, aes(x = Petal.Length, y = Sepal.Length, color = Species)) +
  geom_point() +  # Add the data points
  geom_smooth(method = "lm", se = FALSE, color = "black")  # Add the regression line

summary(step(lm(Sepal.Length~Sepal.Width*Species, data=iris)))

#Here’s an example of how to perform a multiple regression on the iris dataset, using both Petal.Length and Petal.Width to predict Sepal.Length:
model <- lm(Sepal.Length ~ Petal.Length + Petal.Width, data = iris)
# Print the summary of the model
summary(model)

# Create a scatter plot of Sepal.Length vs Petal.Length and add the regression line
ggplot(iris, aes(x = Petal.Length, y = Sepal.Length, color = Species)) +
  geom_point() +  # Add the data points
  geom_smooth(method = "lm", se = FALSE, color = "black") 

# Create a scatter plot of Sepal.Length vs Petal.Width and add the regression line
ggplot(iris, aes(x = Petal.Width, y = Sepal.Length, color = Species)) +
  geom_point() +  # Add the data points
  geom_smooth(method = "lm", se = FALSE, color = "black")  
