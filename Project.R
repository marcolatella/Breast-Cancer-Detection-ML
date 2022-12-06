install.packages("ROCR")
install.packages("pROC ") 
install.packages("caret") 
library("FactoMineR")
library("factoextra")
library(caret)
library(pROC)
library(ROCR)


##### Start Function declaration #####

# Function that extract 2 new covariates from data
expand.data <- function(dataset){
  dataset$volume <- (4/3)*pi*(dataset$mean_radius^3) 
  dataset$compactness <- (dataset$mean_perimeter^2)/(dataset$mean_area-1)
  return(dataset)
}

# Function that move the target covariate to the last position
rearrange.dataset <- function(dataset){
  diagnosis <- dataset$diagnosis
  dataset <- dataset[-6]
  dataset$diagnosis <- diagnosis
  dataset$diagnosis <- ifelse(dataset$diagnosis == 0, "Benign", "Malign")
  return(dataset)
}

# Min Max scaling Function
min.max.scale <- function(dataset){
  for(i in 1:(ncol(dataset)-1)){
    dataset[,i] <- (dataset[,i]-min(dataset[,i]))/(max(dataset[,i])-min(dataset[,i]))
  }
  return(dataset)
}

# Function that split dataset in trainset and testset
split.dataset <- function(dataset){
  ind = sample(2, nrow(dataset), replace = TRUE, prob=c(0.7, 0.3))
  trainset = dataset[ind == 1,]
  testset = dataset[ind == 2,]
  return(list(trainset=trainset,testset=testset))
}

##### End Function Declaration #####



ds <- read.csv("Breast_cancer_data.csv", sep = ",")
ds <- expand.data(ds)
ds <- rearrange.dataset(ds)


### Principal Component Analysis (PCA) ###

ds.active <- ds[, 1:7]
res.pca <- PCA(ds.active, graph = FALSE)

## Stampa degli Autovalori ##
get_eigenvalue(res.pca)

fviz_eig(res.pca, addlabels = TRUE, ylim = c(0, 65))

fviz_pca_var(res.pca)

### End PCA ###



ds.reduced <- ds
ds.reduced <- ds.reduced[-c(1,3,4)]

data <- split.dataset(ds.reduced)
trainset <- data[[1]]
testset <- data[[2]]

testset$diagnosis <- factor(testset$diagnosis)
trainset$diagnosis <- factor(trainset$diagnosis)

#### Dataset Preprocessing for Decision Tree Model

dt.trainset <- trainset
dt.testset <- testset


#### Dataset Preprocessing for Neural Network Model

nn.trainset <- min.max.scale(trainset)
nn.testset <- min.max.scale(testset)


### Neural Network Model ###



fitControl <- trainControl(method = "cv", 
                           number = 10, 
                           classProbs = TRUE, 
                           savePredictions = TRUE,
                           summaryFunction = twoClassSummary)

nnetGrid <-  expand.grid(size = 2,
                         decay = 0.1)

model.nn <- train(diagnosis ~ ., 
               data = nn.trainset,
               method = "nnet",
               metric = "ROC",
               trControl = fitControl,
               tuneGrid = nnetGrid,
               verbose = TRUE)


## Prediction of model on testset
model.nn.probs = predict(model.nn, 
                      nn.testset[,! names(nn.testset) %in% c("diagnosis")], 
                      type = "prob")
## Roc Curve
model.nn.ROC = roc(response = nn.testset[,c("diagnosis")], 
                predictor =model.nn.probs$Malign, 
                levels = levels(nn.testset[,c("diagnosis")]))
## Plot ROC Curve
plot(model.nn.ROC,type="S", col="green")

## Area Under Curve (AUC)
model.nn.ROC


## Single Fold Analysis
model.nn$pred
fold01 <- subset(model.nn$pred, Resample == "Fold01")
confusionMatrix(fold01$pred, fold01$obs, mode = "prec_recall", positive = "Malign")

fold02 <- subset(model.nn$pred, Resample == "Fold02")
confusionMatrix(fold02$pred, fold02$obs, mode = "prec_recall", positive = "Malign")

fold03 <- subset(model.nn$pred, Resample == "Fold03")
confusionMatrix(fold03$pred, fold03$obs, mode = "prec_recall", positive = "Malign")

fold04 <- subset(model.nn$pred, Resample == "Fold04")
confusionMatrix(fold04$pred, fold04$obs, mode = "prec_recall", positive = "Malign")

fold05 <- subset(model.nn$pred, Resample == "Fold05")
confusionMatrix(fold05$pred, fold05$obs, mode = "prec_recall", positive = "Malign")

fold06 <- subset(model.nn$pred, Resample == "Fold06")
confusionMatrix(fold06$pred, fold06$obs, mode = "prec_recall", positive = "Malign")

fold07 <- subset(model.nn$pred, Resample == "Fold07")
confusionMatrix(fold07$pred, fold07$obs, mode = "prec_recall", positive = "Malign")

fold08 <- subset(model.nn$pred, Resample == "Fold08")
confusionMatrix(fold08$pred, fold08$obs, mode = "prec_recall", positive = "Malign")

fold09 <- subset(model.nn$pred, Resample == "Fold09")
confusionMatrix(fold09$pred, fold09$obs, mode = "prec_recall", positive = "Malign")

fold10 <- subset(model.nn$pred, Resample == "Fold10")
confusionMatrix(fold10$pred, fold10$obs, mode = "prec_recall", positive = "Malign")


# Global Confusion Matrix
model.nn.pred.ontestset = c("Benign", "Malign")[apply(model.nn.probs, 1, which.max)]
confusionMatrix(factor(model.nn.pred.ontestset), nn.testset$diagnosis, 
                mode="prec_recall", positive = "Malign")







### Decision Tree Model ###



control <- trainControl(method = "cv", number = 10,
                        classProbs = TRUE,
                        savePredictions = TRUE,
                        summaryFunction = twoClassSummary)


rpart.model= train(diagnosis ~ ., data = dt.trainset, method = "rpart", 
                   metric = "ROC",
                   trControl = control)


## Prediction of model on testset
rpart.probs = predict(rpart.model, 
                      dt.testset[,! names(dt.testset) %in% c("diagnosis")], 
                      type = "prob")


## Roc Curve
rpart.ROC = roc(response = dt.testset[,c("diagnosis")], 
                predictor =rpart.probs$Malign, 
                levels = levels(dt.testset[,c("diagnosis")]))

## Plot ROC Curve
plot(rpart.ROC, type="S", col="blue", add=TRUE)


## Area Under Curve (AUC)
rpart.ROC

 

rpart.model$pred

dt.fold01 <- subset(rpart.model$pred, Resample == "Fold01")
confusionMatrix(dt.fold01$pred, dt.fold01$obs, mode = "prec_recall", positive = "Malign")

dt.fold02 <- subset(rpart.model$pred, Resample == "Fold02")
confusionMatrix(dt.fold02$pred, dt.fold02$obs, mode = "prec_recall", positive = "Malign")

dt.fold03 <- subset(rpart.model$pred, Resample == "Fold03")
confusionMatrix(dt.fold03$pred, dt.fold03$obs, mode = "prec_recall", positive = "Malign")

dt.fold04 <- subset(rpart.model$pred, Resample == "Fold04")
confusionMatrix(dt.fold04$pred, dt.fold04$obs, mode = "prec_recall", positive = "Malign")

dt.fold05 <- subset(rpart.model$pred, Resample == "Fold05")
confusionMatrix(dt.fold05$pred, dt.fold05$obs, mode = "prec_recall", positive = "Malign")

dt.fold06 <- subset(rpart.model$pred, Resample == "Fold06")
confusionMatrix(dt.fold06$pred, dt.fold06$obs, mode = "prec_recall", positive = "Malign")

dt.fold07 <- subset(rpart.model$pred, Resample == "Fold07")
confusionMatrix(dt.fold07$pred, dt.fold07$obs, mode = "prec_recall", positive = "Malign")

dt.fold08 <- subset(rpart.model$pred, Resample == "Fold08")
confusionMatrix(dt.fold08$pred, dt.fold08$obs, mode = "prec_recall", positive = "Malign")

dt.fold09 <- subset(rpart.model$pred, Resample == "Fold09")
confusionMatrix(dt.fold09$pred, dt.fold09$obs, mode = "prec_recall", positive = "Malign")

dt.fold10 <- subset(rpart.model$pred, Resample == "Fold10")
confusionMatrix(dt.fold10$pred, dt.fold10$obs, mode = "prec_recall", positive = "Malign")


# Global Confusion Matrix
rpart.model.pred.ontestset = c("Benign", "Malign")[apply(rpart.probs, 1, which.max)]
confusionMatrix(factor(rpart.model.pred.ontestset), dt.testset$diagnosis, 
                mode="prec_recall", positive = "Malign")



### Confronto Modelli ###


cv.values = resamples(list(nn=model.nn, rpart = rpart.model)) 
summary(cv.values)

## Confronto Intervalli di Confidenza
dotplot(cv.values, metric = "ROC")

bwplot(cv.values, layout = c(3, 1))

splom(cv.values,metric="ROC")

cv.values$timings








