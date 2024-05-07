library(datasets)

ir_data <- iris

head(ir_data)
str(ir_data)
levels(ir_data$Species)

sum(is.na(ir_data))

ir_data <- ir_data[1:100, ]
set.seed(100)
samp = sample(1:100, 80)
ir_test <- ir_data[samp, ]
ir_ctrl <- ir_data[-samp, ]



#install.packages("ggplot2")
#install.packages("GGally")
library(ggplot2)
library(GGally)

ggpairs(ir_test)
ggpairs(ir_ctrl)


y <- ir_test$Species
x <- ir_test$Sepal.Length
glfit <- glm(y~x, family = 'binomial')
summary(glfit)

new_data <- data.frame(x=ir_ctrl$Sepal.Length)
predicted_value <- predict(glfit, new_data, type = "response")
prediction <- data.frame(ir_ctrl$Sepal.Length, ir_ctrl$Species, predicted_value) 
prediction

ggplot(prediction, aes(x = ir_ctrl.Sepal.Length, y = predicted_value, color = ir_ctrl.Species)) +
  geom_point(alpha = 0.6) +  # Adjust point transparency
  labs(title = "Predicted Probability of Species by Sepal Length",
       x = "Sepal Length (cm)",
       y = "Predicted Probability",
       color = "Species") +
  theme_classic() +
  scale_y_continuous(limits = c(0, 1))  # Set y-axis limits between 0 and 1




