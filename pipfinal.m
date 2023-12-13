function [Result,Model,xxTest] = pipfinal(x,xTest,yTest)
%%pipfinal is a function that takes in hyperparameters x, along with
% training and testing data matrices xTrain, yTrain, xTest, and yTest, and
% fits a model of the given type. After completing the model, it finds the
% classification error on the Test data, as well as reporting the processed
% xTest data as xxTest.

global xTrain yTrain

% Normalization
if x(1) == 1
    xxxTrain = normalize(xTrain);
    xxTest = normalize(xTest);
else
    xxxTrain = xTrain;
    xxTest = xTest;
end

% PCA
if x(2) ~= 0
    [~,scoreTrain] = pca(xxxTrain);
    xxxTrain = scoreTrain(:,1:x(2));
    [~,scoreTrain] = pca(xxTest);
    xxTest = scoreTrain(:,1:x(2));
end

if x(3) == 0
    % Optimize tree model
    Model = fitctree(xxxTrain,yTrain,'MinLeafSize',x(4));

elseif x(3) == 1
    % Optimize ensemble model
    if x(8) == 0
        mmeth = 'Bag';
    elseif x(8) == 1
        mmeth = 'AdaBoostM2';
    elseif x(8) == 2
        mmeth = 'RUSBoost';
    end
    Model = fitcensemble(xxxTrain,yTrain,'Method',char(mmeth));

elseif x(3) == 2
    % Optimize knn model

    if x(5) == 0
        distmm = 'cityblock';
    elseif x(5) == 1
        distmm = 'chebychev';
    elseif x(5) == 2
        distmm = 'correlation';
    elseif x(5) == 3
        distmm = 'cosine';
    elseif x(5) == 4
        distmm = 'euclidean';
    elseif x(5) == 5
        distmm = 'hamming';
    elseif x(5) == 6
        distmm = 'jaccard';
    elseif x(5) == 7
        distmm = 'mahalanobis';
    elseif x(5) == 8
        distmm = 'minkowski';
    elseif x(5) == 9
        distmm = 'seuclidean';
    elseif x(5) == 10
        distmm = 'spearman';
    end

    Model = fitcknn(xxxTrain,yTrain,'Distance',char(distmm),'NumNeighbors',x(6),'Standardize',x(7));
end

Result = loss(Model,xxTest,yTest);
end