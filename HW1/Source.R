# Store the data
library(ISLR)
data = Auto

###########################################
#   T A S K   1
###########################################

######
# EX 1
######

# Create linear model
set.seed(42)
model1 = lm(mpg ~ . - name, data = data)

# Results
summary(model1)

######
# EX 2
######

# Sorting the dataset
data.sorted = data[order(data$acceleration), ]

# Polynomial regression
set.seed(42)
m.p1 = lm(mpg ~ poly(acceleration, 1), data = data.sorted)
m.p2 = lm(mpg ~ poly(acceleration, 2), data = data.sorted)
m.p3 = lm(mpg ~ poly(acceleration, 3), data = data.sorted)
m.p4 = lm(mpg ~ poly(acceleration, 4), data = data.sorted)
m.p5 = lm(mpg ~ poly(acceleration, 5), data = data.sorted)

# Plotting the original data
plot(data.sorted$acceleration, data.sorted$mpg,
     main = "Data from the ISLR Auto dataset",
     xlab = "acceleration [s]", ylab = "mpg",
     pch = 19
     )

# Plotting the regression curves
points(data.sorted$acceleration, predict(m.p1), type="l", lwd=5, col="blue")
points(data.sorted$acceleration, predict(m.p2), type="l", lwd=5, col="red")
points(data.sorted$acceleration, predict(m.p3), type="l", lwd=5, col="green")
points(data.sorted$acceleration, predict(m.p4), type="l", lwd=5, col="brown")
points(data.sorted$acceleration, predict(m.p5), type="l", lwd=5, col="orange")

# Creating legend
legend("topleft",
       c("Linear", "Degree 2", "Degree 3", "Degree 4", "Degree 5"),
       col = c("blue", "red", "green", "brown", "orange"),
       lty = c(1,1,1,1,1), lwd = c(5,5,5,5,5)
       )

# Getting adjusted R-squared
round(summary(m.p1)$adj.r.squared, digits = 2)
round(summary(m.p2)$adj.r.squared, digits = 2)
round(summary(m.p3)$adj.r.squared, digits = 2)
round(summary(m.p4)$adj.r.squared, digits = 2)
round(summary(m.p5)$adj.r.squared, digits = 2)


###########################################
#   T A S K   2
###########################################

######
# EX 1
######

# Creating mpg01
med = median(data[,1])
mpg01 = ifelse(data[,1] >= med, 1, 0)
d = data.frame(mpg01, data[,2:9])

# Calculating entropy (using "entropy library")
library(entropy)
probs = prop.table(table(d[,1]))
ent = entropy(probs, unit="log2")
cat("Entropy of the attribute mpg01 is:", ent, "\n")

######
# EX 2
######

# Portion of the data set to be used for training
train_rel = 0.8

# Splitting the dataset
set.seed(42)
indices = sample(nrow(d))
split_index = round(train_rel * nrow(d))
train = d[indices[1:split_index],]
test = d[indices[(split_index + 1):nrow(d)],]

######
# EX 3
######

# Getting the more frequent class of mpg01
train_0 = nrow(subset(train, train$mpg01 == 0))
train_1 = nrow(subset(train, train$mpg01 == 1))
classif = ifelse(train_0 > train_1, 0, 1)

# Computing the accuracy
accuracy = nrow(subset(test, test$mpg01 == classif)) / nrow(test)
cat("Accuracy of the trivial classifier is", round(accuracy, digits=2), "\n")

######
# EX 4
######

# Deleting the 'name' column from both data sets
train_noname = subset(train, select = -name) 
test_noname = subset(test, select = -name)

# Logistic regression
model2 = glm(mpg01 ~ ., 
         data = train_noname, family = binomial(link = "logit"))

# Computing training error rate
p2 = predict.glm(model2, train_noname, type = "response")
y2 = ifelse(p2 > 0.5, 1, 0)
cm2 = table(train_noname[,1], y2)
model2_acc = sum(diag(cm2))/sum(cm2)
cat("Training error rate is:", round(1 - model2_acc, digits = 2), "\n")

# Evaluation of the model on the test set
p2_test = predict.glm(model2, test_noname[,1:8], type = "response")
y2_test = ifelse(p2_test > 0.5, 1, 0)
cm2_test = table(test_noname[,1], y2_test)

# Confusion matrix
cm2_test = table(test_noname[,1], y2_test)
print("Confusion matrix")
cm2_test

# Calculating evaluation metrics
model2_test_acc = sum(diag(cm2_test))/sum(cm2_test)
cat("Test error rate is:", round(1 - model2_test_acc, digits = 2), "\n")
cat("Assuming cars with mpg01 = 1 classified correctly are true positives:\n")
cat("Sensitivity of our model is", round(cm2_test[2,2]/sum(cm2_test[2,]), digits = 2), "\n")
cat("Specificity of our model is", round(cm2_test[1,1]/sum(cm2_test[1,]), digits = 2), "\n")

######
# EX 5
######

# Vector of thresholds
thresholds = c(0.1, 0.3, 0.6, 0.9)

# Computation for each threshold with for loop
for (threshold in thresholds) {
  
  # Computing confusion matrix
  p_loop = predict.glm(model2, test_noname[,1:8], type = "response")
  y_loop = ifelse(p_loop > threshold, 1, 0)
  cm_loop = table(test_noname[,1], y_loop)
  
  # Computing evaluation measures
  precision_loop = cm_loop[2,2]/sum(cm_loop[,2])
  recall_loop = cm_loop[2,2]/sum(cm_loop[2,])
  f_loop = 2 * ((precision_loop * recall_loop)/(precision_loop + recall_loop))
  
  # Printing the evaluation measures
  cat("Confusion matrix for threshold", threshold, "\n")
  print(cm_loop)
  cat("Precision for threshold", threshold, "is", round(precision_loop, digits = 2), "\n")
  cat("Recall for threshold", threshold, "is", round(recall_loop, digits = 2), "\n")
  cat("\n")
}

######
# EX 6
######

library(rpart)
library(rpart.plot)

# Changing mpg01 to factors
train$mpg01 = factor(train$mpg01)
test$mpg01 = factor(test$mpg01)

# Creation of the model
set.seed(42)
m_tree = rpart(mpg01 ~ . - name,
               data = train)

# Plotting the model
rpart.plot(m_tree)

# Computing train error rate
p_tree_train = predict(m_tree, train, type = "class")
cm_t_train = table(train[,1], p_tree_train)
m_tree_train_acc = sum(diag(cm_t_train))/sum(cm_t_train)
cat("Train error rate is:", round(1 - m_tree_train_acc, digits = 2), "\n")

# Computing test error rate
p_tree_test = predict(m_tree, test, type = "class")
cm_t_test = table(test[,1], p_tree_test)
m_tree_test_acc = sum(diag(cm_t_test))/sum(cm_t_test)
cat("Test error rate is:", round(1 - m_tree_test_acc, digits = 2), "\n")

# Creating a deep tree
m_deep = rpart(mpg01 ~ . - name,
               data = train, cp = 0.024609)

# Evaluating best CP
printcp(m_deep)
plotcp(m_deep)
rpart.plot(m_deep)

# Evaluation of error rates for the tree with the best cp
# It's just a copy of what is above
mtb = rpart(mpg01 ~ . - name,
            data = train, cp = 0.024609)

ptb = predict(mtb, train, type = "class")
cmtb = table(train[,1], ptb)
mtb_acc = sum(diag(cmtb))/sum(cmtb)
cat("Train error rate is:", round(1 - mtb_acc, digits = 2), "\n")

ptb = predict(mtb, test, type = "class")
cmtb = table(test[,1], ptb)
mtb_acc = sum(diag(cmtb))/sum(cmtb)
cat("Test error rate is:", round(1 - mtb_acc, digits = 2), "\n")
