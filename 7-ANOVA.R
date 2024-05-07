
# A
time_data <- read.csv("E:/SEM 6/DATA SCIENCE/practice/7a-data.csv")
var.test(time_data$Time_G1, time_data$Time_G2)


# B
sat_level <- read.csv("E:/SEM 6/DATA SCIENCE/practice/7b-data.csv")
anova_table <- aov(sat_level$Sat_index~sat_level$Dept)
summary(anova_table)


# C
dept_exp_data <- read.csv("E:/SEM 6/DATA SCIENCE/practice/7c-data.csv")
anov_table <- aov(formula = Sat_index~Dept + EXP + Dept*EXP, data = dept_exp_data)
summary(anov_table)
