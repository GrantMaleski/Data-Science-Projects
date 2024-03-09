library(tibble) #tidyverse tables
library(dplyr) #tidyverse data manipulation
#install.packages('leaps')
library(leaps)
library(rpart)
library(ggparty)
library(rpart)
#install.packages('GGally')
#install.packages('randomForest')
library(randomForest)
library(caret)
#install.packages('corrplot')
library(corrplot)
#install.packages("heatmaply")
library(heatmaply)


csv_red <- "C:/Users/grant/Downloads/winequality-red.csv"
csv_white <- "C:/Users/grant/Downloads/winequality-white.csv"

# Read the CSV file into a data frame
red_data <- read.csv(csv_red,sep = ";")
white_data <-  read.csv(csv_white,sep = ";")



#give numeric values of actual colour for each dataset
white_data$color <- 0
red_data$color  <- 1


head(red_data)
head(white_data)

#combine the datasets
all_wine <- rbind(red_data, white_data)

#check to see if its linear

# plotting two variables against each other using ggplot2
ggplot(all_wine, aes(x = alcohol, y = quality)) +
  geom_point(color = "blue") +
  labs(title = "Scatter Plot of Alohol vs Quality",
       x = "Alcohol", y = "Qualityl")


#no missing values in dataset
colSums(is.na(all_wine))

summary(all_wine$quality)
summary(white_data$quality)
summary(red_data$quality)

#see dispursement of quality for each quality value
quality_table <- table(all_wine$quality)

#convert the table to a data frame
quality_df <- as.data.frame(quality_table)

quality_df_transposed <- t(quality_df)


(quality_df_transposed)

#distribution of quality
quality_table= table(all_wine$quality)

quality_table

barplot(quality_table, main = "Quality Score Distribution",
        xlab = "Quality Score", ylab = "Number of Observations",
        col = "skyblue", border = "black")


#investigate outliers
Q1 <- quantile(all_wine$quality, 0.25)
Q3 <- quantile(all_wine$quality, 0.75)
IQR <- Q3 - Q1

#define the lower and upper bounds for outliers
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR

#identify rows with quality values outside the bounds
outliers <- which(all_wine$quality < lower_bound | all_wine$quality > upper_bound)

outliers

#remove rows with values 3,4,8,9
all_wine <- all_wine[!(all_wine$quality %in% c(3, 4, 8, 9)), ]


#get the correlation matrix
correlation_matrix <- cor(numeric_variables)
correlation_matrix

#print the correlation matrix
print(correlation_matrix)
heatmaply(correlation_matrix,
          main = "Correlation Heatmap",
          width = 800,
          height = 600,
          fontsize_row = 12,
          fontsize_col = 12,
)

#isolate pred variables
prediction_variables <- names(all_wine)[!(names(all_wine) %in% c("quality", "color"))]
pred_variables <- paste(prediction_variables, collapse = " + ")

pred_variables

#create formula for easy creation of models for all pred variables
formula <- as.formula(paste("quality ~", paste(names(all_wine)[1:11], collapse = " + ")))






#-------Classify if it is red or white----------

wine_quality <- rpart(formula, data = all_wine)


#this checks the overlap of two of the most polarized variables
ggplot(all_wine, aes(x = volatile.acidity, y = total.sulfur.dioxide, color = factor(color))) +
  geom_point(aes(shape = factor(color))) +
  labs(x = "Volatile Acidity", y = "Total Sulfur Dioxide", title = "Scatter Plot: Volatile Acidity vs Total Sulfur Dioxide") +
  scale_color_manual(values = c("pink", "red")) +
  theme_minimal()



#--- best split
BSplit <- all_wine %>%
  mutate(pred_split = case_when(
    color == 1 ~ "red",
    color == 0 ~ "white"))


#highest score variables
wine_class <- rpart(data = BSplit,
                    pred_split ~ volatile.acidity + total.sulfur.dioxide + chlorides,
                    method = "class",
                    control = rpart.control(cp = 0.01)) 

#display the decision tree
autoplot(as.party(wine_class))


printcp(wine_class)

all_wine_class <- rpart(data = BSplit,
                    pred_split ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + sulphates + alcohol,
                    method = "class",
                    control = rpart.control(cp = 0.01)) 

#display the decision tree
autoplot(as.party(all_wine_class))


printcp(all_wine_class)

#choosing the three strategic variables improves cross validation by .01
opt = which.min(wine_quality$cptable[,"xerror"])
cp = wine_quality$cptable[opt, "CP"]
cp

cat_cp = wine_class$cptable %>%
  as.data.frame %>%
  slice(which.min(xerror)) %>%
  select(CP) %>%
  as.numeric


pruned_class_model = prune(wine_class,cp)
pruned_class_model
printcp(pruned_class_model)
autoplot(as.party(pruned_class_model))


predictions <- predict(pruned_class_model, newdata = all_wine, type = "class")

unique(predictions1)
unique(all_wine$quality)

#create a new dataset with the predicted value

pred_wine <- all_wine

#add a new column 'pred_color' based on the predictions
pred_wine$pred_color <- ifelse(predictions == "red", 1, 0)

pred_wine <- pred_wine[, c("pred_color", setdiff(names(pred_wine), "pred_color"))]

misidentified <- sum(pred_wine$color) - sum(pred_wine$pred_color)
misidentified
#misidentified 39 colors
100 - misidentified / nrow(pred_wine)



#-------Regression tree without prediction color-------------

# best subset selection for all varaibles
pred_best_model <- regsubsets(formula, data = all_wine, nvmax = 11)

pred_formula <- as.formula(paste("quality ~", paste(names(all_wine)[1:12], collapse = " + ")))


pred_best_model
pred_best_summary = summary(pred_best_model)
which.max(best_summary$adjr2)
print(coef(best_model,11))

#build the rpart model using the formula and data = all_wine
wine_quality <- rpart(pred_formula, data = all_wine)

plot(as.party(wine_quality))
printcp(wine_quality)

mean(red_data$quality)
mean(white_data$quality)



opt = which.min(wine_quality$cptable[,"xerror"])
cp = wine_quality$cptable[opt, "CP"]
cp

pruned_quality = prune(wine_quality, cp)
pruned_quality
plot(as.party(pruned_quality))

subset_data <- all_wine[1, 1:11]

subset_data %>% mutate(pred_quality = predict(pruned_quality, newdata = subset_data))



library(caret)
fitControl = trainControl(method = "cv",number = 10)




rpartFit = train(pred_formula,
                 data = all_wine,
                 method = "rpart",
                 trControl = fitControl)


plot(as.party(rpartFit$finalModel))



rpartFit





#-------Regression tree with the predicted color-------------


#build the rpart model using the formula and data = all_wine
pred_wine_quality <- rpart(formula, data = all_wine)



plot(as.party(pred_wine_quality))
printcp(pred_wine_quality)




opt = which.min(pred_wine_quality$cptable[,"xerror"])
pred_cp = pred_wine_quality$cptable[opt, "CP"]
cp



pred_pruned_quality = prune(pred_wine_quality, pred_cp)
pred_pruned_quality
plot(as.party(pred_pruned_quality))



subset_data <- all_wine[1, 1:12]

subset_data %>% mutate(pred_quality = predict(pred_pruned_quality, newdata = subset_data))




library(caret)
fitControl = trainControl(method = "cv",number = 10)


pred_rpartFit = train(pred_formula,
                 data = pred_wine,
                 method = "rpart",
                 trControl = fitControl)




plot(as.party(pred_rpartFit$finalModel))

pred_rpartFit


- #---------- Random Forests-----


#start clean with original data and predictor variables
clean_wine_data <- all_wine[1:12]

#sample indices for the test set
ind <- sample(1:nrow(clean_wine_data), size = 50)



#test and train sets
test_wine <- clean_wine_data[ind, ]
train_wine <- clean_wine_data[-ind, ]

#convert 'quality' to a factor
train_wine$quality <- as.factor(train_wine$quality)
test_wine$quality <- as.factor(test_wine$quality)


formula <- quality ~ volatile.acidity + total.sulfur.dioxide + chlorides + alcohol


#random forest model
rf <- randomForest(formula, data = train_wine)

#predict on the test set
pred <- predict(rf, newdata = test_wine)



confusionMatrix(pred, test_wine$quality)



