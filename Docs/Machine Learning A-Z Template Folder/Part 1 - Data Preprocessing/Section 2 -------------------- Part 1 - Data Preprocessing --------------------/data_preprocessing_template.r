dataset = read.csv("./Data.csv")
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)


# Encoding

dataset$Country = factor(dataset$Country,
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1,2,3))


dataset$Purchased = factor(dataset$Purchased,
                         levels = c('Yes', 'No'),
                         labels = c(1,2))

# Splitting the dataset

#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
trainig_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Feature Scaling

trainig_set[, 2:3] = scale(trainig_set[, 2:3])
test_set[, 2:3] = scale(test_set[, 2:3])

