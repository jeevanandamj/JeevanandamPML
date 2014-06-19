plotUserFeature <- function(user, feature, data){
  userData <- data[data$user_name==user, ]
  qplot(x=userData$X, y=userData[, feature], colour=userData$classe, xlab="X", ylab=paste(user, feature))
}

plotFeature <- function(feature, data){
  qplot(x=data$X, y=data[, feature], colour=data$classe, xlab="X", ylab=paste(feature))
}

plotFeatureImp <- function(feature, data){
  qplot(y=data[, feature], colour=data$classe, xlab="SeqAlong", ylab=paste(feature))
}

stripDerivedColumns <- function(data)
{
  subset(data, select=c("roll_belt", "pitch_belt", "yaw_belt", "total_accel_belt", "gyros_belt_x", "gyros_belt_y", "gyros_belt_z", "accel_belt_x", "accel_belt_y", "accel_belt_z", "magnet_belt_x", "magnet_belt_y", "magnet_belt_z", "roll_arm", "pitch_arm", "yaw_arm", "total_accel_arm", "gyros_arm_x", "gyros_arm_y", "gyros_arm_z", "accel_arm_x", "accel_arm_y", "accel_arm_z", "magnet_arm_x", "magnet_arm_y", "magnet_arm_z", "roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell", "gyros_dumbbell_x", "gyros_dumbbell_y", "gyros_dumbbell_z", "accel_dumbbell_x", "accel_dumbbell_y", "accel_dumbbell_z", "magnet_dumbbell_x", "magnet_dumbbell_y", "magnet_dumbbell_z", "roll_forearm", "pitch_forearm", "yaw_forearm", "gyros_forearm_x", "gyros_forearm_y", "gyros_forearm_z", "accel_forearm_x", "accel_forearm_y", "accel_forearm_z", "magnet_forearm_x", "magnet_forearm_y", "magnet_forearm_z", "classe"))
}

stripLowImportanceColumns <- function(data)
{
  result <- data
  result <- subset(data, select=-pitch_dumbbell)
  result <- subset(data, select=-accel_belt_y)
  result <- subset(data, select=-total_accel_belt)
  result <- subset(data, select=-magnet_arm_x)
  result <- subset(data, select=-gyros_belt_x)
  result <- subset(data, select=-magnet_arm_y)
  result
}

performFoldCrossValidation <- function(trainingData, k) {
  folds <- createFolds(trainingData$classe, k = k, list = T)
  lapply(seq_along(folds), function(index) {
    testingIdx <- folds[[index]]
    trainingIdx <- as.vector(c(folds[-index], recursive=T))
    testing <- trainingData[testingIdx, ]
    training <- trainingData[trainingIdx, ]
    
    testing <- stripDerivedColumns(testing)
    training <- stripDerivedColumns(training)
    
    rfFitAllFeatures <- randomForest(classe ~ ., 
                             data=training, 
                             na.action=na.fail,
                             ntree = 501) #avoid random tie breaking
    rfFitImportantFeatures <- randomForest(classe ~ ., 
                             data=stripLowImportanceColumns(training), 
                             na.action=na.fail,
                             ntree = 501) #avoid random tie breaking
                             
    testPrediction <- predict(rfFitAllFeatures, testing)
    confMDef <- confusionMatrix(testPrediction, testing$classe)
    testPrediction <- predict(rfFitImportantFeatures, stripLowImportanceColumns(testing))
    confMImp <- confusionMatrix(testPrediction, testing$classe)
    c(all=confMDef, cut=confMImp)
  })
}

compareAccuracies <- function(performFoldCrossValidationResultList) {
  accuraciesDef <- sapply(performFoldCrossValidationResultList, function(pairDefImp) {
    pairDefImp$all.overall[[1]]
  })
  accuraciesImp <- sapply(performFoldCrossValidationResultList, function(pairDefImp) {
    pairDefImp$cut.overall[[1]]
  })
  
  cbind(all=accuraciesDef , cut=accuraciesImp)
}

customPartition <- function(data, p){
  
  idxDataSortedUserClass <- order(data$user_name, data$classe, data$X)
  dataSortedUserClass<-data[idxDataSortedUserClass, ]
  
  idxPartition <- lapply(levels(data$user_name), function(username) {
    dataUserLog <- dataSortedUserClass$user_name == username
    
    dataUser <- dataSortedUserClass[dataUserLog, ]
    
    idxDataUser <- idxDataSortedUserClass[dataUserLog]
    
    idxDataUserClassList <- lapply(levels(data$classe), function(class) {
      dataUserClassLog <- dataUser$classe == class
      
      idxDataUserClass <- idxDataUser[dataUserClassLog]
      
      dataUserClassLength <- sum(dataUserClassLog)
      
      trainingSetSize <- as.integer(round(dataUserClassLength * p, 0))
      testingSetStartIndex <- trainingSetSize+1
      
      trainingSet <- idxDataUserClass[1:trainingSetSize]
      
      #testingSet <- idxDataUserClass[testingSetStartIndex:dataUserClassLength]
      
      #return(list(train=trainingSet, test=testingSet))
      return(trainingSet)
    })
    
    #idxTrainingUser <- collapseIntoVector(idxDataUserClassList, isTrain=T)
    idxTrainingUser <- c(idxDataUserClassList, recursive=T)
    
    #idxTestingUser <- collapseIntoVector(idxDataUserClassList, isTrain=F)
    
    #return(list(train=idxTrainingUser, test=idxTestingUser))
    return(idxTrainingUser)
  })
  
  #idxTraining <- collapseIntoVector(idxPartition, isTrain=T)
  idxTraining <- c(idxPartition, recursive=T)
  #idxTesting <- collapseIntoVector(idxPartition, isTrain=F)
  
  return(idxTraining)
}

collapseIntoVector <- function(idxPairedList, isTrain) {
  idxResult <- c(lapply(idxPairedList, function(pair) {
    if (isTrain) {
      return(pair$train)
    }else {
      return(pair$test)
    }
  }), recursive=T)
  return(idxResult)
}