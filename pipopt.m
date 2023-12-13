function [Result,Model] = pipopt(x)
%%pipopt is a function that takes in the optimization hyperparameters x and
% the training data xTrain and yTrain as two matrices in order to fit a
% 5-fold cv model of the given type to said data, and to report model and
% cv results

global xTrain yTrain

% Normalization
if x(1) == 1
    xxxTrain = normalize(xTrain);
else
    xxxTrain = xTrain;
end

% PCA
if x(2) ~= 0
    [~,scoreTrain] = pca(xxxTrain);
    xxxTrain = scoreTrain(:,1:x(2));
end

if x(3) == 0
    % Optimize tree model
    Model = fitctree(xxxTrain,yTrain,'KFold',5,'MinLeafSize',x(4));

elseif x(3) == 1
    % Optimize ensemble model
    if x(8) == 0
        mmeth = 'Bag';
    elseif x(8) == 1
        mmeth = 'AdaBoostM2';
    elseif x(8) == 2
        mmeth = 'RUSBoost';
    end
    Model = fitcensemble(xxxTrain,yTrain,'KFold',5,'Method',char(mmeth));

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

    Model = fitcknn(xxxTrain,yTrain,'KFold',5,'Distance',char(distmm),'NumNeighbors',x(6),'Standardize',x(7));
end

Result = kfoldLoss(Model);
end