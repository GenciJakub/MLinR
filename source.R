library(ISLR)
library(gbm)
library(randomForest)
d = Auto
d$origin = as.factor(d$origin)

#####################
#### T A S K   1 ####
#####################

# Getting the best depth
const_depth_error = numeric(0)
for (depth in 1:10) {
    model = gbm(mpg ~ cylinders + displacement + horsepower + weight +
                      acceleration + year + origin,
               data = d, distribution = "gaussian",
               n.trees = 500, shrinkage = 0.01, interaction.depth = depth, cv.folds = 8)
    
    const_depth_error = c(const_depth_error, mean((model$cv.fitted - d$mpg)^2))
}
print(const_depth_error)
plot(const_depth_error, type = "l", xlab="Depth", ylab="MSE with 500 trees")

# Depth and number fo trees
error = numeric(0)
for (ntrees in 1:10) {
  for (depth in 1:10) {
    model = gbm(mpg ~ cylinders + displacement + horsepower + weight +
                  acceleration + year + origin,
                data = d, distribution = "gaussian",
                n.trees = ntrees*100, shrinkage = 0.01, interaction.depth = depth, cv.folds = 8)
    
    error = c(error, mean((model$cv.fitted - d$mpg)^2))
  }
}  

# Create matrix of errors
# Constant number of trees in one row, constatnt depth in one column
error_matrix = matrix(error, nrow=10, byrow=TRUE)

# Plotting
plot(error_matrix[,1], type = "l", xlab="Number of trees [hundreds]", ylab="Mean sqaured error", col="black")
lines(error_matrix[,3], col="blue")
lines(error_matrix[,5], col="green")
lines(error_matrix[,7], col="purple")
lines(error_matrix[,9], col="red")
legend(x="topright", legend=c("Depth 1", "Depth 3", "Depth 5", "Depth 7", "Depth 9"),
       fill=c("black","blue","green","purple","red"))


#####################
#### T A S K   2 ####
#####################

# Number of observations in leaves
error = numeric(0)
for (minN in 1:20) {
  model = gbm(mpg ~ cylinders + displacement + horsepower + weight +
                acceleration + year + origin,
              data = d, distribution = "gaussian",
              n.trees = 1000, shrinkage = 0.01, interaction.depth = 5, cv.folds = 8, n.minobsinnode = minN)
  
  error = c(error, mean((model$cv.fitted - d$mpg)^2))
}
print(error)
# Best is 6

# Crossval for random forests
cv.sample <- sample(1:nrow(d))
add.zeros <- 8 - nrow(d) %% 8
if(add.zeros < 8) cv.sample <- c(cv.sample, rep(0, add.zeros))
cv.index <- matrix(data = cv.sample, nrow = 8)

# Testing nodesize in random forests
nodesize_error = numeric(0)
error = numeric(0)
for (minN in 1:20) {
  for (fold in 1:8) {
    cv.train <- d[ - cv.index[fold,][cv.index[fold,] > 0], ]
    cv.test  <- d[ cv.index[fold,][cv.index[fold,] > 0], ]
    
    RFmodel = randomForest(mpg ~ cylinders + displacement + horsepower + weight +
                              acceleration + year + origin,
                            data = cv.train, ntree = 1000, nodesize = minN)
    
    predictions = predict(RFmodel, cv.test)
    nodesize_error = c(nodesize_error, (predictions - cv.test$mpg)^2)
  }
  error = c(error,mean(nodesize_error))
}
print(error)
# Best is 1

# Best boosting model
best_boost_model = gbm(mpg ~ cylinders + displacement + horsepower + weight +
              acceleration + year + origin,
            data = d, distribution = "gaussian",
            n.trees = 1000, shrinkage = 0.01, interaction.depth = 5, cv.folds = 8, n.minobsinnode = 6)
boost_error = mean((model$cv.fitted - d$mpg)^2)
print(boost_error)

# Best random forest
nodesize_error = numeric(0)
for (fold in 1:8) {
  cv.train <- d[ - cv.index[fold,][cv.index[fold,] > 0], ]
  cv.test  <- d[ cv.index[fold,][cv.index[fold,] > 0], ]
  
  RFmodel = randomForest(mpg ~ cylinders + displacement + horsepower + weight +
                           acceleration + year + origin,
                         data = cv.train, ntree = 1000, nodesize = 1)
  
  predictions = predict(RFmodel, cv.test)
  nodesize_error = c(nodesize_error, (predictions - cv.test$mpg)^2)
}
error = mean(nodesize_error)
print(error)
