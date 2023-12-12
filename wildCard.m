%% Wildcard Project  - Hyperparameter Search Space Exploration
% William Baumchen
close all; clear; clc

%% Data Inport

% Import Data
load('pipelineobs.mat')
% Output XTrace Data of optimization
Xtrace = results.XTrace;

%% Plotting

% Create p-c plot of decision tree model evaluations
figure(1)
indm = Xtrace.solver == 0;
tab0 = Xtrace(indm,1:4);
tab0(:,3) = [];
res0 = results.ObjectiveTrace(indm);
Evaluation = res0;
pcoor0 = parallelplot(sortrows([tab0,table(Evaluation)],2));
pcoor0.GroupVariable = 'featureNum';
title('Decision Tree Plot')

% Create p-c plot of ensemble model evaluations
figure(2)
indm = Xtrace.solver == 1;
tab1 = Xtrace(indm,:);
tab1(:,3:7) = [];
res1 = results.ObjectiveTrace(indm);
Evaluation = res1;
pcoor1 = parallelplot(sortrows([tab1,table(Evaluation)]));
pcoor1.GroupVariable = 'featureNum';
title('Classification Ensemble Plot')

% Create p-c plot of knn model evaluations
figure(3)
indm = Xtrace.solver == 2;
tab2 = Xtrace(indm,:);
tab2(:,8) = [];
tab2(:,3:4) = [];
res2 = results.ObjectiveTrace(indm);
Evaluation = res2;
pcoor2 = parallelplot(sortrows([tab2,table(Evaluation)]));
pcoor2.GroupVariable = 'featureNum';
title('knn Plot')
