# Importing the dataset
dataset = read.csv('50_Startups.csv')

# Encoding categorical data
dataset$State = factor(dataset$State,
                         levels = c('New York', 'California', 'Florida'),
                         labels = c(1, 2, 3))
# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


#Fitting Multiple Linear Regrassion to the Training set
regressor = lm(formula = Profit ~ .,
               data = training_set)
summary(regressor)

regressor2 = lm(formula = Profit ~ R.D.Spend,
               data = training_set)
summary(regressor2)

regressor3 = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = training_set)
summary(regressor3)

regressor4 = lm(formula = Profit ~ R.D.Spend + Marketing.Spend + State,
                data = training_set)
summary(regressor4)

y_pred = predict(regressor, newdata = test_set)

y_pred2 = predict(regressor2, newdata = test_set)

y_pred3 = predict(regressor3, newdata = test_set)

y_pred4 = predict(regressor4, newdata = test_set)