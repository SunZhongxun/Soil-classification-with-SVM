%%Empty environment variables
warning off;
close all;
clear;
clc;

%%Import data
filePath = 'D:\文献\土壤分类\数据\SVM_4+1_new.xlsx';
trainingData = readtable(filePath);
inputTable = trainingData;
predictorNames = {'qc', 'fs'}; % Select the predictor variables 'qc' and 'fs'.
predictors = inputTable(:, predictorNames);
response = inputTable.type;
isCategoricalPredictor = [false, false];
classNames = {'1', '2', '3', '4'};

% Disrupt training data
randomIndices = randperm(height(trainingData));
trainingData = trainingData(randomIndices, :);

predictors = trainingData(:, predictorNames);
% Before using fitcecoc, make sure that the response variable is also an array of strings or cells
response = string(trainingData.type);

%%Normalization process
% Extract maximum and minimum values from training data
minValues = min(table2array(predictors));
maxValues = max(table2array(predictors));
% Normalize the training data
normalizedPredictors = (table2array(predictors) - minValues) ./ (maxValues - minValues);
% Converting an array back to a table
normalizedPredictors = array2table(normalizedPredictors, 'VariableNames', predictorNames);
% Update predictors to normalized data
predictors = normalizedPredictors;
%% Mapping of data
figure;
predictorsArray = table2array(predictors);
gscatter(predictorsArray(:,1), predictorsArray(:,2), response);
title('Data distribution');
xlabel('qc');
ylabel('fs');
hold on;

% Train the classifier
template = templateSVM(...
    'KernelFunction', 'linear', ...
    'PolynomialOrder',[], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'Solver', 'ISDA');% 使用SMO、L1QP、ISDA优化算法
classificationSVM = fitcecoc(...
    predictors, ...
    response, ...
    'Learners', template, ...
    'Coding', 'onevsone', ...
    'ClassNames', classNames);


% Plotting raw data points
figure;
gscatter(predictorsArray(:,1), predictorsArray(:,2), response);
hold on;

% Create a grid to cover the data range
[xGrid, yGrid] = meshgrid(linspace(min(predictorsArray(:,1)), max(predictorsArray(:,1)), 100), ...
                         linspace(min(predictorsArray(:,2)), max(predictorsArray(:,2)), 100));

% Predictions for each grid point
labels = predict(classificationSVM, [xGrid(:), yGrid(:)]);
% Reshape labels to the same size as the xGrid and yGrid.
labels = reshape(labels, size(xGrid));

% For each pair of categories, plot the decision boundary
desiredPairs = {[1, 2], [2, 3], [3, 4]};  % Specify the category pairs you wish to display
for pair = desiredPairs
    i = pair{1}(1);
    j = pair{1}(2);
    contour(xGrid, yGrid, (strcmp(labels, classNames{i})) | (strcmp(labels, classNames{j})), [1 1], 'k', 'LineWidth', 2);
end

title('Data distribution and Decision boundaries');
xlabel('qc');
ylabel('fs');
hold off;

% Creating Result Structures with Prediction Functions
predictorExtractionFcn = @(t) t(:, predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

% Adding fields to the result structure
trainedClassifier.RequiredVariables = {'Rf', 'fs', 'qc'};
trainedClassifier.ClassificationSVM = classificationSVM;
trainedClassifier.About = '此结构体是从分类学习器 R2023a 导出的训练模型。';
trainedClassifier.HowToPredict = sprintf('要对新表 T 进行预测，请使用: \n [yfit,scores] = c.predictFcn(T) \n将 ''c'' 替换为作为此结构体的变量的名称，例如 ''trainedModel''。\n \n表 T 必须包含由以下内容返回的变量: \n c.RequiredVariables \n变量格式(例如矩阵/向量、数据类型)必须与原始训练数据匹配。\n忽略其他变量。\n \n有关详细信息，请参阅 <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>。');

% Cross-validation using the training set
partitionedModel = crossval(classificationSVM, 'KFold', 5);

% Compute cross-validated predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Calculate the confusion matrix
[C, order] = confusionmat(response, validationPredictions, 'Order', classNames);

% Plotting the confusion matrix for cross-validation
figure;
confusionchart(C, order);
title('Confusion matrix for cross-validation');

% Calculating the accuracy of cross-validation
crossvalAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
% Calculate Precision, Recall, F1-score for each category.
numClasses = numel(classNames);
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1score = zeros(numClasses, 1);

for i = 1:numClasses
    % true positives
    TP = C(i, i);
    % false positives
    FP = sum(C(:, i)) - TP;
    % false negatives
    FN = sum(C(i, :)) - TP;

    precision(i) = TP / (TP + FP);
    recall(i) = TP / (TP + FN);
    f1score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
end

% Calculate average Precision, Recall, F1-score
avgPrecision = mean(precision);
avgRecall = mean(recall);
avgF1Score = mean(f1score);

% Display precision, recall, f1score and their averages in a MATLAB table
convertedClassNames = cellfun(@num2str, classNames, 'UniformOutput', false);

if isrow(convertedClassNames)
    convertedClassNames = convertedClassNames';
end

rowNames = [convertedClassNames; {'Average'}];
resultsTable = array2table([precision, recall, f1score; avgPrecision, avgRecall, avgF1Score], ...
                           'VariableNames', {'Precision', 'Recall', 'F1_Score'}, ...
                           'RowNames', rowNames);

% Display Forms
disp(resultsTable);

% Calculate TPR and FPR for each category on the training set

numLabels = numel(classNames);
allFPR_train = cell(numLabels, 1);
allTPR_train = cell(numLabels, 1);
allAUC_train = zeros(numLabels, 1);

% ====== ROC of the training set ======
% Using Training Response Data
binResponses_train = double(response);

figure; % Creating a new graphic
hold on;

% Calculating scores using training data
[~, trainScores] = predict(classificationSVM, predictors);  % 注意这里，我们需要得分

for i = 1:numLabels
    % Get the score of the current category as a positive category
    scores = trainScores(:, i);

    % Marks the current category as 1 and all others as 0
    binResponse = double(strcmp(response, classNames{i}));

    % Calculate TPR, FPR, and AUC
    [X, Y, ~, AUC] = perfcurve(binResponse, scores, 1);

    % Plot the ROC curve for the current category
    plot(X, Y, 'LineWidth', 2, 'DisplayName', sprintf('Class %s (AUC = %.2f)', classNames{i}, AUC));
end

% Setting Graphic Properties
title('训练集的ROC for multi-class classification');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
legend('show', 'Location', 'SouthEast');
grid on;
hold off;

% ====== 1. Read the new validation dataset ======
newValidationFilePath = 'D:\文献\土壤分类\数据\场地5.xlsx'; % Modify here to point to your new Excel file
newValidationData = readtable(newValidationFilePath);
% Disrupting the new validation dataset
randomIndicesValidation = randperm(height(newValidationData));
newValidationData = newValidationData(randomIndicesValidation, :);

newValidationPredictors = newValidationData(:, predictorNames);
newValidationResponse = newValidationData.type;

% Normalize the new validation dataset using the maximum and minimum values previously extracted from the training data
normalizedNewValidationPredictors = (table2array(newValidationPredictors) - minValues) ./ (maxValues - minValues);
% Converting an array back to a table
normalizedNewValidationPredictors = array2table(normalizedNewValidationPredictors, 'VariableNames', predictorNames);
% Update newValidationPredictors to normalized data
newValidationPredictors = normalizedNewValidationPredictors;

% ====== 2. Prediction using already trained models ======
[newPredictions, newScores] = predict(classificationSVM, newValidationPredictors);

% ====== 3. Calculating accuracy, confusion matrices, and other metrics ======
% Calculate the confusion matrix
% Convert newValidationResponse to an array of strings.
newValidationResponseString = string(newValidationResponse);

[C_new, order_new] = confusionmat(newValidationResponseString, newPredictions);

% ====== 4. Calculate Precision, Recall, F1-score for each category for new validation set ======
precision_new = zeros(numClasses, 1);
recall_new = zeros(numClasses, 1);
f1score_new = zeros(numClasses, 1);

for i = 1:numClasses
    % true positives
    TP = C_new(i, i);
    % false positives
    FP = sum(C_new(:, i)) - TP;
    % false negatives
    FN = sum(C_new(i, :)) - TP;

    precision_new(i) = TP / (TP + FP);
    recall_new(i) = TP / (TP + FN);
    f1score_new(i) = 2 * (precision_new(i) * recall_new(i)) / (precision_new(i) + recall_new(i));
end

% Calculate the average Precision, Recall, F1-score for the new validation set
avgPrecision_new = mean(precision_new);
avgRecall_new = mean(recall_new);
avgF1Score_new = mean(f1score_new);

% Display the precision, recall, and f1score of the new validation set, along with their mean values, in a MATLAB table
% 1. Convert classNames to an array of cells as strings.
convertedClassNames = cellfun(@num2str, classNames, 'UniformOutput', false);

% 2. Check if the converted class name is a row vector, if so, convert it to a column vector
if isrow(convertedClassNames)
    convertedClassNames = convertedClassNames';
end

% 3. Constructing RowNames
rowNames = [convertedClassNames; {'Average'}];

% 4. Creating a Table with RowNames and Your Calculated Data
resultsTable_new = array2table([precision_new, recall_new, f1score_new; avgPrecision_new, avgRecall_new, avgF1Score_new], ...
                               'VariableNames', {'Precision', 'Recall', 'F1_Score'}, ...
                               'RowNames', rowNames);

% Display the results table for the new validation set
disp('Evaluation metrics for new validation sets:');
disp(resultsTable_new);

% Plot the confusion matrix for the new validation set
figure;
confusionchart(C_new, order_new);
title('Confusion matrix for new validation sets');

% Calculate the accuracy of the new validation set
newTestAccuracy = sum(newPredictions == newValidationResponseString) / length(newValidationResponseString);
disp(['Accuracy of the new validation set: ', num2str(newTestAccuracy)]);

% Calculate TPR and FPR for each category for the new validation set
numLabels = numel(classNames);
allFPR_new = cell(numLabels, 1);
allTPR_new = cell(numLabels, 1);
allAUC_new = zeros(numLabels, 1);

% ROC for the new validation set
figure;
hold on;

% Using the new validation response data
binResponses_new = string(newValidationResponse);

for i = 1:numLabels
    % Get the score of the current category as a positive category
    scores = newScores(:, i);

    % Marks the current category as 1 and all others as 0
    binResponse = double(strcmp(binResponses_new, classNames{i}));

    % Calculate TPR, FPR, and AUC
    [X, Y, ~, AUC] = perfcurve(binResponse, scores, 1);

    % Plot the ROC curve for the current category
    plot(X, Y, 'LineWidth', 2, 'DisplayName', sprintf('Class %s (AUC = %.2f)', classNames{i}, AUC));
end

% Setting Graphic Properties
title('ROC for multi-class classification of new validation sets');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
legend('show', 'Location', 'SouthEast');
grid on;
hold off;