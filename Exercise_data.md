# Practical Machine Learning Course Project
Marsela Yulita  
14 September 2017  



## Introduction
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

#### Load the data

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(rattle)
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.1.0 Copyright (c) 2006-2017 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```
## 
## Attaching package: 'rattle'
```

```
## The following object is masked from 'package:randomForest':
## 
##     importance
```

```r
#Download the data
if(!file.exists("pml-training.csv")){
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "pml-training.csv")}

if(!file.exists("pml-testing.csv")){
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "pml-testing.csv")}

#Read the data
trainingData<- read.csv("pml-training.csv", sep=",", header=TRUE, na.strings = c("NA","",'#DIV/0!'))
testingData<- read.csv("pml-testing.csv", sep=",", header=TRUE, na.strings = c("NA","",'#DIV/0!'))

dim(trainingData)
```

```
## [1] 19622   160
```

```r
dim(testingData)
```

```
## [1]  20 160
```

#### Preprocessing data

```r
#Remove fields with NA
trainingData <- trainingData[,(colSums(is.na(trainingData)) == 0)]
testingData <- testingData[,(colSums(is.na(testingData)) == 0)]

dim(trainingData)
```

```
## [1] 19622    60
```

```r
dim(testingData)
```

```
## [1] 20 60
```

```r
#Preprocess step
numericalcol <- which(lapply(trainingData, class) %in% "numeric")

preprocessModel <-preProcess(trainingData[,numericalcol],method=c('knnImpute', 'center', 'scale'))
pre_trainingData <- predict(preprocessModel, trainingData[,numericalcol])
pre_trainingData$classe <- trainingData$classe

pre_testingData <-predict(preprocessModel,testingData[,numericalcol])

#Remove NearZeroVariance variables
nzv <- nearZeroVar(pre_trainingData,saveMetrics=TRUE)
pre_trainingData <- pre_trainingData[,nzv$nzv==FALSE]

nzv <- nearZeroVar(pre_testingData,saveMetrics=TRUE)
pre_testingData <- pre_testingData[,nzv$nzv==FALSE]
```

#### Train Data
I used the random-forest technique to generate a predictive model, It turned out showed a good performance.


```r
#create data partition
set.seed(2017)
dtTrain<- createDataPartition(pre_trainingData$classe, p=0.6, list=FALSE)
training<- pre_trainingData[dtTrain, ]
validation <- pre_trainingData[-dtTrain, ]
dim(training) ; dim(validation)
```

```
## [1] 11776    28
```

```
## [1] 7846   28
```

```r
#train model with random forest
modelFit <- train(classe ~., method="rf", data=training, trControl=trainControl(method='cv'), number=5, allowParallel=TRUE, importance=TRUE )
modelFit
```

```
## Random Forest 
## 
## 11776 samples
##    27 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 10598, 10599, 10598, 10600, 10598, 10598, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9913381  0.9890418
##   14    0.9900633  0.9874305
##   27    0.9859872  0.9822762
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

#### Evaluate the model (out-of-sample error)

```r
predictvalid <- predict(modelFit, validation)
validmatrix <- confusionMatrix(validation$classe, predictvalid)
validmatrix
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2227    4    0    0    1
##          B   15 1496    7    0    0
##          C    0   12 1345   11    0
##          D    0    0   23 1262    1
##          E    2    1    1    3 1435
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9897          
##                  95% CI : (0.9872, 0.9918)
##     No Information Rate : 0.286           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9869          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9924   0.9888   0.9775   0.9890   0.9986
## Specificity            0.9991   0.9965   0.9964   0.9963   0.9989
## Pos Pred Value         0.9978   0.9855   0.9832   0.9813   0.9951
## Neg Pred Value         0.9970   0.9973   0.9952   0.9979   0.9997
## Prevalence             0.2860   0.1928   0.1754   0.1626   0.1832
## Detection Rate         0.2838   0.1907   0.1714   0.1608   0.1829
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9958   0.9926   0.9870   0.9927   0.9988
```

The accuracy of the prediction is 99.07%. Hence, the out-of-sample error is 100% - 99.07% = 0.93%.

#### Using the model with the test data

```r
pred_final <- predict(modelFit, pre_testingData)
pred_final
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

Here are the results, we will use them for the submission of this course project in the coursera platform.
