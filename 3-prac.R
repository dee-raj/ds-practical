# One Sample T-Test : https://www.statology.org/one-sample-t-test/
# Paired T-Test : https://www.statology.org/paired-samples-t-test/
# Two Sample T-Test : https://www.statology.org/two-sample-t-test/

#one sample
csv_data <- read.csv("E:/SEM 6/DATA SCIENCE/practice/3-onesample.csv")
df <- as.data.frame(csv_data)
t.test(df$Time,alternative = "greater",mu=90)

#Two sample
csv_data <- read.csv("E:/SEM 6/DATA SCIENCE/practice/3-twosample.csv")
df <- as.data.frame(csv_data)
t.test(df$Time1,df$Time2)
# alternative = ("two.sided" it means not equal to) is default and var.equal="True" is used to let t test know that variance is equal and do not need to calculate variance

#Paired
csv_data <- read.csv("E:/SEM 6/DATA SCIENCE/practice/3-pairedsample.csv")
df <- as.data.frame(csv_data)
t.test(df$Time_Before,df$Time_After,alternative = "less",paired = TRUE)

#The choice between "less" or "greater" in the `alternative` argument of the `t.test` function depends on the specific hypothesis you are testing. 
#- If you believe that the average time after is **less** than the average time before, you should use `alternative = "less"`. 
#- If you believe that the average time after is **greater** than the average time before, you should use `alternative = "greater"`.

#In your case, since you stated that your alternative hypothesis (Ha) is that the average time after is less than before, you should use `alternative = "less"`. 

#Here's how you can interpret the results:
#- If the `p-value` is less than your chosen significance level (commonly 0.05), you would reject the null hypothesis and conclude that the average time after is less than before (if `alternative = "less"`) or greater than before (if `alternative = "greater"`).
#- If the `p-value` is greater than your chosen significance level, you would fail to reject the null hypothesis and conclude that there is not enough evidence to say the average time after is less than before (or greater than before, depending on your alternative hypothesis).
