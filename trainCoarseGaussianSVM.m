function [trainedClassifier, validationAccuracy] = trainCoarseGaussianSVM(trainingData,x,t)
inputTable = trainingData;
predictorNames = {'MajorAxisLength', 'MinorAxisLength', 'ConvexArea','Solidity','Circularity'};
predictors = inputTable(:, predictorNames);
response = inputTable.Ground_truth;
isCategoricalPredictor = [false, false, false];

classificationSVM = fitcsvm(...
    predictors, ...
    response, ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 6.9, ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', ['C' 'a' 'n' 'c' 'e' 'r'; 'N' 'o' 'r' 'm' 'a' 'l']);

% Create the result struct with predict function

predictorExtractionFcn = @(t) t(:, predictorNames);
svmPredictFcn = @(x) predict(classificationSVM,x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = {'MajorAxisLength', 'MinorAxisLength', 'ConvexArea','Solidity','Circularity'};
trainedClassifier.ClassificationSVM = classificationSVM;

inputTable = trainingData;
predictorNames = {'MajorAxisLength', 'MinorAxisLength', 'ConvexArea','Solidity','Circularity'};
predictors = inputTable(:, predictorNames);
response = inputTable.Ground_truth;
isCategoricalPredictor = [false, false, false];

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 5);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');

% Compute validation predictions and scores
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);
