%%Empty environment variables
warning off;
close all;
clear;
clc;

%%Import data
filePath = 'D:\文献\土壤分类\数据\SVM_4+1_new.xlsx';
trainingData = readtable(filePath);

% Select the predictor variables 'qc' and 'fs'.
predictorNames = {'qc', 'fs'};
predictors = trainingData(:, predictorNames);
response = string(trainingData.type);

%% Data distribution map
figure;
predictorsArray = table2array(predictors);
gscatter(predictorsArray(:,1), predictorsArray(:,2), response);
title('数据分布');
xlabel('qc');
ylabel('fs');
hold on;

% Normalization process
minValues = min(table2array(predictors));
maxValues = max(table2array(predictors));
normalizedPredictors = (table2array(predictors) - minValues) ./ (maxValues - minValues);
normalizedPredictors = array2table(normalizedPredictors, 'VariableNames', predictorNames);
predictors = normalizedPredictors;

%% Get all unique categories
classNames = unique(response);
de_pop = 60;%Population size
de_Maxiter = 30;%Maximum number of iterations
F = 0.5;% Perturbation factor
CR = 0.8;% Crossover probability
dim = 2;
lb = [1e-3, 1e-3];
ub = [1e3, 1e3];
Sol = rand(de_pop, dim);
Fitness = zeros(1, de_pop);

for i = 1:de_pop
    x = Sol(i, :);
    boxConstraint = x(1);
    kernelScale = x(2);

    % SVM training
    template = templateSVM('KernelFunction', 'linear', 'PolynomialOrder', [],...
                           'KernelScale', kernelScale, 'BoxConstraint', boxConstraint, ...
                           'Standardize', true, 'Solver', 'ISDA');
    mdl = fitcecoc(predictors, response, 'Learners', template, 'Coding', 'onevsone');
    partitionedModel = crossval(mdl, 'KFold', 5);
    errorRate = kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
    Fitness(i) = errorRate;
end

for time = 1:de_Maxiter
    for i = 1:de_pop
        % Variation and crossover
        r = randperm(de_pop, 5);
        mutantpos = Sol(r(1),:) + F * (Sol(r(2),:) - Sol(r(3),:)) + F * (Sol(r(4),:) - Sol(r(5),:));
        jj = randi(dim);
        for d = 1:dim
            if rand() < CR || d == jj
                crossoverpos(d) = mutantpos(d);
            else
                crossoverpos(d) = Sol(i,d);
            end
        end
        % Checking for transgressions
        crossoverpos(crossoverpos > ub) = ub(crossoverpos > ub);
        crossoverpos(crossoverpos < lb) = lb(crossoverpos < lb);

        boxConstraint = crossoverpos(1);
        kernelScale = crossoverpos(2);

        % SVM training
        template = templateSVM('KernelFunction', 'linear', 'PolynomialOrder', [],...
                               'KernelScale', kernelScale, 'BoxConstraint', boxConstraint, ...
                               'Standardize', true, 'Solver', 'ISDA');
        mdl = fitcecoc(predictors, response, 'Learners', template, 'Coding', 'onevsone');
        partitionedModel = crossval(mdl, 'KFold', 5);
        new_fitness = kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');

        if new_fitness < Fitness(i)
            Sol(i,:) = crossoverpos;
            Fitness(i) = new_fitness;
        end
    end
    [de_best, bestindex] = min(Fitness);  % Update bestindex
    disp(['第' num2str(time), '代:' num2str(de_best)]);
end

bestBoxConstraint = Sol(bestindex, 1);
bestKernelScale = Sol(bestindex, 2);

%% Training the final SVM model with optimal parameters
template = templateSVM(...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [],...
    'KernelScale', bestKernelScale, ...
    'BoxConstraint', bestBoxConstraint, ...
    'Standardize', true, ...
    'Solver', 'ISDA');

classificationSVM = fitcecoc(...
    predictors, ...
    response, ...
    'Learners', template, ...
    'Coding', 'onevsone');
% Show results
disp(['Best BoxConstraint: ', num2str(bestBoxConstraint)]);
disp(['Best KernelScale: ', num2str(bestKernelScale)]);
[de_best, ~] = min(Fitness);
disp(['Minimum cross-validation error rate: ', num2str(de_best)]);

%% Create a grid to cover the data range
[xGrid, yGrid] = meshgrid(linspace(min(predictorsArray(:,1)), max(predictorsArray(:,1)), 100), ...
                         linspace(min(predictorsArray(:,2)), max(predictorsArray(:,2)), 100));

% Plotting raw data points
figure;
gscatter(predictorsArray(:,1), predictorsArray(:,2), response);
hold on;

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

title('Data Distribution and Decision Boundaries');
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
binResponses_train = string(response);

figure; % Creating a new graphic
hold on;

% Calculating scores using training data
[~, trainScores] = predict(classificationSVM, predictors);  

for i = 1:numLabels
    % Get the score of the current category as a positive category
    scores = trainScores(:, i);

    % Marks the current category as 1 and all others as 0
    binResponse = strcmp(binResponses_train, classNames{i});

    % Calculate TPR, FPR, and AUC
    [X, Y, ~, AUC] = perfcurve(binResponse, scores, 1);

    % Development of ROC curves for the current category
   plot(X, Y, 'LineWidth', 2, 'DisplayName', sprintf('Class %s (AUC = %.2f)', classNames{i}, AUC));
end

% Setting graphic properties
title('ROC for multi-class classification on training set');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
legend('show', 'Location', 'SouthEast');
grid on;
hold off;

% ====== 1. Read the new validation dataset ======
newValidationFilePath = 'D:\文献\土壤分类\数据\场地5.xlsx'; % Modify here to point to your new Excel file
newValidationData = readtable(newValidationFilePath);
% Extracting new predictor variables
newValidationPredictors = newValidationData(:, predictorNames);
% Extracting new response variables
newValidationResponse = string(newValidationData.type);  % Assuming the column name is 'type'
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
[C_new, order_new] = confusionmat(newValidationResponse, newPredictions);

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

% 4. Creating a table with RowNames and your calculated data
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
newTestAccuracy = sum(newPredictions == newValidationResponse) / length(newValidationResponse);
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
    binResponse = strcmp(binResponses_new, classNames{i});

    % Calculate TPR, FPR, and AUC
    [X, Y, ~, AUC] = perfcurve(binResponse, scores, 1);

    % Plot the ROC curve for the current category
    plot(X, Y, 'LineWidth', 2, 'DisplayName', sprintf('Class %s (AUC = %.2f)', classNames{i}, AUC));
end

% Setting graphic properties
title('ROC for multi-class classification of new validation sets');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
legend('show', 'Location', 'SouthEast');
grid on;
hold off;
