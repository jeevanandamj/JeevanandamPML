Weight Lifting Exercises Prediction
========================================================

## Purpose

This document describes the analisys I made when solving the Practical Machine Learning course project.

The purpose of the assignment is to predict how well a weight lifting exercise is performed from a known set of common mistakes. As predictors, I use the raw data provided by four sensors. For more information about the data set please see http://groupware.les.inf.puc-rio.br/har. The variable to predict is the *classe* variable of the data set. Its values range from "A" to "E". "A" is the correct way to do the exercise while "B", "C", "D" and "E" represent common mistakes made by athletes.

### Assumptions
Having no more information about the dataset, and to simplify the analysis, I assume that:
* The athletes do not get too tired after some repetitions. That means the last points collected from a given exercise have the same weight in terms of usefull information as the first points.
* With the exception of the trainer, each athlete is not influenced by the way other athletes make each exercise.

## Feature extraction and selection

By plotting most of the features, I reached to the following conclusions about the training set:
* Individuals perform each exercise a bit differently, so it will be dificult to predict for a new individual.
* The derived features such as mean, variance, standard deviation, max, min, amplitude, kurtosis and skewness did not seem to add much information about each classe and it reduced the amount of information available for training. It compresses the data but, due to the fact the we only have six young men performing the exercise, I believe using as much data as possible to perform the training yields a better predictor.
* The "E" classe mistake will be easier to distinguish from the other mistakes.
* The "B" and "A" classes wll be hard to tell apart.

In light of these conclusions, I removed all the derived features except for the roll, pitch and yaw.


```r
source('./helpers.R')
trainingData <- stripDerivedColumns(read.csv("pml-training.csv"))
colnames(trainingData)
```

```
##  [1] "roll_belt"         "pitch_belt"        "yaw_belt"         
##  [4] "total_accel_belt"  "gyros_belt_x"      "gyros_belt_y"     
##  [7] "gyros_belt_z"      "accel_belt_x"      "accel_belt_y"     
## [10] "accel_belt_z"      "magnet_belt_x"     "magnet_belt_y"    
## [13] "magnet_belt_z"     "roll_arm"          "pitch_arm"        
## [16] "yaw_arm"           "total_accel_arm"   "gyros_arm_x"      
## [19] "gyros_arm_y"       "gyros_arm_z"       "accel_arm_x"      
## [22] "accel_arm_y"       "accel_arm_z"       "magnet_arm_x"     
## [25] "magnet_arm_y"      "magnet_arm_z"      "roll_dumbbell"    
## [28] "pitch_dumbbell"    "yaw_dumbbell"      "gyros_dumbbell_x" 
## [31] "gyros_dumbbell_y"  "gyros_dumbbell_z"  "accel_dumbbell_x" 
## [34] "accel_dumbbell_y"  "accel_dumbbell_z"  "magnet_dumbbell_x"
## [37] "magnet_dumbbell_y" "magnet_dumbbell_z" "roll_forearm"     
## [40] "pitch_forearm"     "yaw_forearm"       "gyros_forearm_x"  
## [43] "gyros_forearm_y"   "gyros_forearm_z"   "accel_forearm_x"  
## [46] "accel_forearm_y"   "accel_forearm_z"   "magnet_forearm_x" 
## [49] "magnet_forearm_y"  "magnet_forearm_z"  "classe"
```


## Cross-Validation

Given the nature of the assignment, I used a random forest with 501 trees to predict the *classe* variable.

I used a random forest in 80% of the training data to understand the most important features:


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
set.seed(2343)
inTrain <- createDataPartition(trainingData$classe, p=0.8, list=F)
training <- trainingData[inTrain, ]
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
rfFit <- randomForest(classe ~ ., 
                    data=training, 
                    na.action=na.fail,
                    ntree = 501, #avoid random tie breaking
                    importance=T)
importance(rfFit)[order(importance(rfFit)[, 6]), ]
```

```
##                       A     B     C     D     E MeanDecreaseAccuracy
## pitch_dumbbell    12.16 19.23 17.11 13.28 14.70                17.08
## magnet_arm_x      16.33 15.32 18.56 17.93 14.91                17.70
## total_accel_belt  14.77 18.40 14.37 14.32 16.27                18.46
## accel_belt_y      13.00 16.84 14.00 15.82 13.94                18.70
## magnet_arm_y      13.94 18.35 19.39 21.55 14.87                19.15
## accel_arm_x       17.48 20.15 20.58 21.60 16.44                21.35
## gyros_belt_y      12.68 17.44 17.49 14.82 17.78                21.70
## gyros_arm_z       13.39 14.06 15.89 13.84 12.88                21.85
## magnet_forearm_x  16.95 20.43 21.35 17.79 19.89                22.62
## gyros_forearm_x   12.02 18.12 17.05 17.43 14.38                23.56
## accel_belt_x      15.22 19.22 18.76 15.36 15.83                24.07
## pitch_arm         16.34 22.60 21.12 20.98 18.22                24.10
## total_accel_arm   11.11 22.43 20.92 19.77 19.86                24.49
## accel_dumbbell_x  17.84 25.53 20.35 19.66 21.72                24.83
## gyros_dumbbell_y  21.26 22.68 26.94 22.06 20.16                25.98
## accel_arm_z       13.51 23.97 23.70 23.03 20.17                26.49
## accel_belt_z      19.65 25.38 22.17 22.58 21.58                26.83
## accel_forearm_x   19.23 26.46 26.50 29.76 22.40                27.14
## accel_arm_y       20.01 23.28 18.49 20.42 17.66                27.35
## gyros_belt_x      18.38 18.21 21.31 14.71 18.28                27.46
## gyros_forearm_y   15.97 24.67 23.90 23.72 20.74                27.54
## yaw_forearm       21.09 21.89 21.57 19.29 22.26                28.04
## magnet_belt_y     20.99 26.14 26.22 24.81 23.80                28.11
## gyros_arm_x       17.71 27.03 23.16 24.22 22.69                28.48
## magnet_forearm_y  22.06 23.76 26.00 21.34 21.69                28.63
## magnet_arm_z      22.31 26.25 24.61 20.19 19.11                29.04
## accel_forearm_y   20.99 22.15 22.64 20.30 24.19                29.28
## roll_forearm      29.64 25.76 32.29 23.45 24.69                29.56
## magnet_dumbbell_x 24.72 28.94 31.37 29.25 22.77                30.09
## yaw_dumbbell      20.55 27.26 24.99 24.26 26.15                31.06
## gyros_belt_z      20.59 28.83 27.50 22.07 27.74                31.23
## accel_forearm_z   19.68 25.16 24.79 24.04 24.70                31.32
## yaw_arm           23.53 27.24 25.29 27.26 22.00                31.63
## roll_dumbbell     24.50 28.05 29.43 30.60 27.58                32.18
## gyros_dumbbell_x  18.11 25.39 21.77 20.97 20.51                32.41
## magnet_belt_z     25.17 28.35 24.22 30.62 25.61                32.66
## magnet_belt_x     17.23 27.73 27.31 23.01 26.04                33.82
## accel_dumbbell_z  21.92 28.32 26.72 27.99 33.65                35.03
## roll_arm          22.23 33.06 28.43 28.52 22.82                35.08
## gyros_forearm_z   17.00 26.78 23.25 17.98 18.05                35.63
## magnet_forearm_z  26.70 31.59 28.25 28.08 26.15                35.95
## gyros_arm_y       20.29 31.31 25.73 26.18 20.42                36.62
## gyros_dumbbell_z  18.39 25.43 18.44 20.62 17.52                38.09
## accel_dumbbell_y  28.61 30.03 35.19 30.07 31.67                39.01
## pitch_forearm     32.49 35.22 39.47 34.58 34.48                41.41
## magnet_dumbbell_y 35.16 36.90 45.20 36.64 34.34                43.20
## pitch_belt        33.28 52.72 40.32 35.34 33.60                50.49
## magnet_dumbbell_z 45.04 40.89 51.93 37.52 37.71                52.02
## roll_belt         43.16 49.47 47.14 48.99 44.77                59.05
## yaw_belt          50.16 45.77 44.08 47.12 36.23                65.45
##                   MeanDecreaseGini
## pitch_dumbbell              139.19
## magnet_arm_x                191.72
## total_accel_belt            165.08
## accel_belt_y                101.63
## magnet_arm_y                181.27
## accel_arm_x                 211.49
## gyros_belt_y                 92.78
## gyros_arm_z                  50.32
## magnet_forearm_x            183.19
## gyros_forearm_x              62.56
## accel_belt_x                 95.53
## pitch_arm                   145.11
## total_accel_arm              82.17
## accel_dumbbell_x            212.91
## gyros_dumbbell_y            204.41
## accel_arm_z                 106.12
## accel_belt_z                322.53
## accel_forearm_x             270.73
## accel_arm_y                 126.51
## gyros_belt_x                 79.67
## gyros_forearm_y             109.19
## yaw_forearm                 141.51
## magnet_belt_y               300.03
## gyros_arm_x                 111.81
## magnet_forearm_y            175.50
## magnet_arm_z                151.44
## accel_forearm_y             116.32
## roll_forearm                492.62
## magnet_dumbbell_x           407.16
## yaw_dumbbell                209.55
## gyros_belt_z                251.70
## accel_forearm_z             193.39
## yaw_arm                     196.39
## roll_dumbbell               332.24
## gyros_dumbbell_x             96.99
## magnet_belt_z               324.04
## magnet_belt_x               203.05
## accel_dumbbell_z            288.41
## roll_arm                    262.93
## gyros_forearm_z              67.51
## magnet_forearm_z            240.89
## gyros_arm_y                 114.55
## gyros_dumbbell_z             66.97
## accel_dumbbell_y            367.75
## pitch_forearm               629.52
## magnet_dumbbell_y           564.13
## pitch_belt                  571.30
## magnet_dumbbell_z           638.15
## roll_belt                  1024.29
## yaw_belt                    736.24
```

I used 10-fold cross validation to decide whether to take out the features *pitch_dumbbell*, *accel_belt_y*, *total_accel_belt*, *magnet_arm_x*, *magnet_arm_y* or not:



```r
set.seed(2343)
confMFoldList <- performFoldCrossValidation(trainingData, 10)
summary(compareAccuracies(confMFoldList))
```

```
##       all             cut       
##  Min.   :0.994   Min.   :0.995  
##  1st Qu.:0.996   1st Qu.:0.996  
##  Median :0.996   Median :0.996  
##  Mean   :0.997   Mean   :0.997  
##  3rd Qu.:0.997   3rd Qu.:0.997  
##  Max.   :1.000   Max.   :1.000
```

As you can see, I can fit a random forest without those features and don't loose much accuracy. On the other hand, the random forest fit will be less sensitive to changes in the training set. Also, the expected out-of-sample error is very low.

## Validation

The final forest fit parameters are shown bellow:

```r
set.seed(2343)
trainingData <- stripLowImportanceColumns(trainingData)
rfFit <- randomForest(classe ~ ., 
                       data=trainingData, 
                       na.action=na.fail,
                       ntree = 501)
print(rfFit)
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = trainingData, ntree = 501,      na.action = na.fail) 
##                Type of random forest: classification
##                      Number of trees: 501
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.3%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 5577    2    0    0    1   0.0005376
## B   10 3783    4    0    0   0.0036871
## C    0    9 3410    3    0   0.0035067
## D    0    0   20 3195    1   0.0065299
## E    0    0    2    6 3599   0.0022179
```

And its evaluation on the testing data:

```r
testingData <- read.csv("pml-testing.csv")
testingPrediction <- predict(rfFit, testingData)
testingPrediction
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```


