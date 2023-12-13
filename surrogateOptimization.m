%% Surrogate Optimization
% William Baumchen
close all; clear; clc

% Display style
verboze = 'iter';

%% Data Inport
% Set global coordinates for use in functions
global xTrain yTrain

% Import Data
datain = readmatrix("winequality-white.csv");
% Shuffle Data Entries for Splitting Data
% Set random seed for reproducibility
% rng(47)
% rng(54)
rng(42)
datain = datain(randperm(size(datain,1)),:);
% Set Fraction of Entries for Test Set
a = 0.2;
% Split Data
xTest = datain(1:round(a*size(datain,1)),1:11);
yTest = datain(1:round(a*size(datain,1)),12);
xTrain = datain(round(a*size(datain,1))+1:end,1:11);
yTrain = datain(round(a*size(datain,1))+1:end,12);

%% Pipeline Optimization

% Hyperparameters for optimization
% 'normVal',[0,1]
% 'featureNum',[0,11]
% 'solver',[0,2]
% 'minLeaf',[1,max(2,height(xTrain)-1)]
% 'distance',[0,10]
% 'numNeigh',[1,max(2,round(height(xTrain)/2))]
% 'knStandard',[0,1]
% 'Method',[0,2]

% Create function
fun = @(x)pipopt(x);
% Assemble hyperparameter upper and lower bounds
lb = [0,0,0,2,0,1,0,0];
ub = [1,11,2,max(2,height(xTrain)-1),10,max(2,round(height(xTrain)/2)),1,2];
% Specify variables as integers
intcon = 1:8;
% Set optimization options
options = optimoptions('surrogateopt','PlotFcn','surrogateoptplot','Display',char(verboze));
% Run Surrogate Optimization
[x,fval,exitflag,output,trials] = surrogateopt(fun,lb,ub,intcon,options)

%% Evaluate Trained Model

% Get optimal model and evaluation, and parameters
[Eval,Model,xxTest] = pipfinal(x,xTest,yTest);

%% Save Model Workspace
save('surrgopt1.mat')
