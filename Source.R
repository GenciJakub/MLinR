setwd("C:/Users/Jakub/Documents/Vyska/LS2122/MLvR/DUe1/")

# Q1
titanic_train = read.csv("train.csv", header=TRUE, sep=',')
titanic_test = read.csv("test.csv", header=TRUE, sep=',')

# remove 'Cabin' column - mostly empty and mostly unique
titanic_train = subset(titanic_train, select = -Cabin)
titanic_test = subset(titanic_test, select = -Cabin)

# replace NA values in 'Age' column by mean
titanic_train$Age[is.na(titanic_train$Age)] = mean(titanic_train$Age, na.rm=TRUE)
titanic_test$Age[is.na(titanic_test$Age)] = mean(titanic_test$Age, na.rm=TRUE)

# merging the data
titanic_test$Survived = 2
merged_data = rbind(titanic_test, titanic_train)

plot(merged_data[,c(2,4,5,6,7,8,9,10)])
plot(merged_data[,c(5,8,9)], col=factor(merged_data$Survived)) #green = 2 - BIAS!

# Q2
# Test set doesn't have 'Survived' data, therefore we can't evaluate the classifier on it
titanic_train2 = titanic_train

# convert survived into factor => survived = 1
titanic_train2$Survived = factor(titanic_train2$Survived, levels = c(0,1))

# Setting seed just for reproducibility
set.seed(42)
indices = sample(nrow(titanic_train))
split_index = round(0.9 * nrow(titanic_train))  # just to simplify
train_train = titanic_train2[indices[1:split_index],]
train_test = titanic_train2[indices[(split_index + 1):nrow(titanic_train2)],]

# Generating the model
m1 = glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, 
         data = train_train, family = binomial(link = "logit")) 
# Getting from predictions (log odds) to confusion matrix
p1 = predict.glm(m1, train_test, type = "response")
y1 = ifelse(p1 > 0.5, 1, 0)
cm1 = table(train_test[,2], y1)

# Evaluation  of the model
cm1
cat("Accuracy of the model is", sum(diag(cm1))/sum(cm1),"\n")
cat("Precision of the model is", cm1[2,2]/sum(cm1[,2]), "\n")
cat("Recall of the model is", cm1[2,2]/sum(cm1[2,]), "\n")
cat("F-measure of the model is", 2*((cm1[2,2]/sum(cm1[,2]) * cm1[2,2]/sum(cm1[2,]))/(cm1[2,2]/sum(cm1[,2]) + cm1[2,2]/sum(cm1[2,]))),"\n")

# Q3
movies_big = read.csv("mov.development.csv", header=TRUE, sep='\t')
# movies = read.csv("mov.development.csv", header=TRUE, sep='\t')
movies = movies_big[1:2500,]
movies = subset(movies, select = -zip)

# splitting into train and test set
set.seed(42)
indices = sample(nrow(movies))
split_index = round(0.9 * nrow(movies))  # just to simplify
movies_train = movies[indices[1:split_index],]
movies_test = movies[indices[(split_index + 1):nrow(movies)],]

# Splitting training set into 5 subsets with each having rating as a 2 level factor
m_tr1 = movies_train
m_tr1$rating = factor(m_tr1$rating, levels = c(1,2,3,4,5), labels = c(1,0,0,0,0))

m_tr2 = movies_train
m_tr2$rating = factor(m_tr2$rating, levels = c(1,2,3,4,5), labels = c(0,1,0,0,0))

m_tr3 = movies_train
m_tr3$rating = factor(m_tr3$rating, levels = c(1,2,3,4,5), labels = c(0,0,1,0,0))

m_tr4 = movies_train
m_tr4$rating = factor(m_tr4$rating, levels = c(1,2,3,4,5), labels = c(0,0,0,1,0))

m_tr5 = movies_train
m_tr5$rating = factor(m_tr5$rating, levels = c(1,2,3,4,5), labels = c(0,0,0,0,1))

# Fitting models on the train set
m1_1 = glm(rating ~ . - timestamp - title - release_date - imdb_url,
           data = m_tr1, family = binomial(link = "logit"))

m1_2 = glm(rating ~ . - timestamp - title - release_date - imdb_url,
           data = m_tr2, family = binomial(link = "logit"))

m1_3 = glm(rating ~ . - timestamp - title - release_date - imdb_url,
           data = m_tr3, family = binomial(link = "logit"))

m1_4 = glm(rating ~ . - timestamp - title - release_date - imdb_url,
           data = m_tr4, family = binomial(link = "logit"))

m1_5 = glm(rating ~ . - timestamp - title - release_date - imdb_url,
           data = m_tr5, family = binomial(link = "logit"))


# Getting predictions
p1_1 = predict.glm(m1_1, movies_test, type = "response")
p1_2 = predict.glm(m1_2, movies_test, type = "response")
p1_3 = predict.glm(m1_3, movies_test, type = "response")
p1_4 = predict.glm(m1_4, movies_test, type = "response")
p1_5 = predict.glm(m1_5, movies_test, type = "response")
p1_all = matrix(c(p1_1, p1_2, p1_3, p1_4, p1_5), nrow = 5)
classifications = apply(p1_all, MARGIN = 2, FUN = which.max)
classifications

# Creating confusion matrix
cm1 = table(movies_test[,3], classifications)

# Function for computing evaluation parameters
get_stats = function(cm, wanted_rating){
  TP = cm[wanted_rating, wanted_rating]
  TN = sum(cm[-wanted_rating, -wanted_rating])
  FP = sum(cm[,wanted_rating]) - TP
  FN = sum(cm[wanted_rating,]) - TP
  
  p = TP/(TP+FP)
  r = TP/(TP+FN)
  
  cat("Accuracy of the model", wanted_rating  ,"is", (TP+TN)/(sum(cm)) ,"\n")
  cat("Precision of the model", wanted_rating  ,"is", p, "\n")
  cat("Recall of the model", wanted_rating  ,"is", r, "\n")
  cat("F-measure of the model", wanted_rating  ,"is", 2*((p * r)/(p + r)),"\n")
  cat("\n")
}

# Getting evaluation
for(val in 1:5)
{
  get_stats(cm1, val)
}