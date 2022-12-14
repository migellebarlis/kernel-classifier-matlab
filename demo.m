clear;clc;close all;

% Load the data as an image datastore using the imageDatastore
% function and specify the folder containing the image data.
imds = imageDatastore("data", ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames", ...
    FileExtensions='.png');

% Determine the number of classes in the training data.
classes = categories(imds.Labels);
numClasses = numel(classes);
numImgs = numel(imds.Files);

% Read all images from the image datastore to a cell array
ims = reshape(readall(imds),[numImgs/numClasses numClasses]);

% Remove the means from all the images
for i = 1:numel(ims)
    ims{i} = ims{i} - mean(ims{i}(:));
end

% Divide the images for training, validation, and testing
train = ims(1:60,:);
valid = ims(61:70,:);
test = ims(71:80,:);

% The network used in this example requires input images of size
% 128-by-128.
inputSize = [128 128 1];

% Define the network for image classification.
layers = [
    imageInputLayer(inputSize, Normalization='none', Name='input')
    convolution2dLayer(5,16, Name='feat_1') % 124 x 124
    reluLayer(Name='relu_1')
    maxPooling2dLayer(2,Stride=[2 2],Name='pool_1') % 62 x 62
    convolution2dLayer(3,32, Name='feat_2') % 60 x 60
    reluLayer(Name='relu_2')
    maxPooling2dLayer(2,Stride=[2 2],Name='pool_2') % 30 x 30
    convolution2dLayer(3,64, Name='feat_3') % 28 x 28
    reluLayer(Name='relu_3')
    maxPooling2dLayer(2,Stride=[2 2],Name='pool_3')]; % 14 x 14

% Create a dlnetwork object from the layer array.
net = dlnetwork(layers);

% Train for n epochs
numEpochs = 100;

% Initialize the training progress plot.
figure
C = colororder;
lineLossTrain = animatedline(Color=C(2,:));
xlabel("Epoch")
ylabel("Loss")
grid on

% Initialize the gradient parameter for the ADAM solver.
averageGrad = [];
averageSqGrad = [];

% Train the network using a custom training loop.
start = tic;

% Loop over epochs.
for epoch = 1:numEpochs
    % Preprocess the training data
    X = preprocessTrainingData(train,inputSize);

    % Evaluate the model gradients, state, and loss using dlfeval and the
    % modelLoss function and update the network state.
    [gradients,state] = dlfeval(@modelLoss,net,X);
    net.State = state;

    % Update the network parameters using the ADAM optimizer.
    [net,averageGrad,averageSqGrad] = adamupdate(net,gradients,averageGrad,averageSqGrad,epoch);

    % Preprocess the validation data
    X = preprocessTrainingData(valid,inputSize);

    % Validate the model by getting the loss
    validLoss = modelValidate(net,X);
    
    % Display the training progress.
    D = duration(0,0,toc(start),Format="hh:mm:ss");
    validLoss = double(validLoss);
    addpoints(lineLossTrain,epoch,validLoss)
    title("Epoch: " + epoch + ", Elapsed: " + string(D))
    drawnow
end

% Preprocess the test data
X = preprocessTestData(valid,inputSize);

% Test the model
testLoss = modelTest(net,X)';

% Visualize the loss.
comb = nchoosek(1:8,2);
figure
scatter(1:28,testLoss',"filled")
xlabel("Kernel Combination Index")
ylabel("Loss")
title("Test Loss Summary")
legend(strcat('Image',arrayfun(@num2str,(1:10)','UniformOutput',false)))

function [gradients,state] = modelLoss(net,X)
% Forward data through network.
[Y,state] = forward(net,X);

numFeat = numel(Y(:,:,:,1));
numBatch = size(Y,4)/2;

Y = reshape(Y,[numFeat numBatch*2]);
Y1 = Y(:,1:numBatch);
Y2 = Y(:,numBatch+1:end);

% Create penalties for feature vectors approaching zero
lambda = 1;
p1 = lambda * exp(-(sum(Y1 .* Y1) .^ 2) ./ 0.01);
p2 = lambda * exp(-(sum(Y2 .* Y2) .^ 2) ./ 0.01);
p = sum(p1 + p2);

% Calculate normalised dot product.
loss = sum((sum(Y1 .* Y2) .^ 2) ./ (sum(Y1 .* Y1 + Y2 .* Y2) + eps)) + p;

% Calculate gradients of loss with respect to learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end

function loss = modelValidate(net,X)
% Forward data through network.
Y = predict(net,X);

numFeat = numel(Y(:,:,:,1));
numBatch = size(Y,4)/2;

Y = reshape(Y,[numFeat numBatch*2]);
Y1 = Y(:,1:numBatch);
Y2 = Y(:,numBatch+1:end);

% Create penalties for feature vectors approaching zero
lambda = 1;
p1 = lambda * exp(-(sum(Y1 .* Y1) .^ 2) ./ 0.01);
p2 = lambda * exp(-(sum(Y2 .* Y2) .^ 2) ./ 0.01);
p = sum(p1 + p2);

% Calculate normalised dot product.
loss = sum((sum(Y1 .* Y2) .^ 2) ./ (sum(Y1 .* Y1 + Y2 .* Y2) + eps)) + p;

end

function loss = modelTest(net,X)
% Make prediction.
Y = predict(net,X);

numFeat = numel(Y(:,:,:,1));
numBatch = size(Y,4)/8;

Y = reshape(Y,[numFeat numBatch 8]);
loss = zeros([numBatch 8]);
comb = nchoosek(1:8,2);

for i = 1:size(comb,1)
    Y1 = squeeze(Y(:,:,comb(i,1)));
    Y2 = squeeze(Y(:,:,comb(i,2)));

    % Create penalties for feature vectors approaching zero
    lambda = 1;
    p1 = lambda * exp(-(sum(Y1 .* Y1) .^ 2) ./ 0.01);
    p2 = lambda * exp(-(sum(Y2 .* Y2) .^ 2) ./ 0.01);
    p = p1 + p2;
    
    % Calculate normalised dot product.
    loss(:,i) = (((sum(Y1 .* Y2) .^ 2) ./ (sum(Y1 .* Y1 + Y2 .* Y2) + eps)) + p)';
end

end

function X = preprocessTrainingData(I,inputSize)
numBatch = size(I,1);
numClass = size(I,2);

X1 = zeros([inputSize numBatch]);
X2 = zeros([inputSize numBatch]);
Xsel = randperm(numBatch);

for i = 1:numBatch
    Csel = randperm(numClass,2);
    X1(:,:,:,i) = im2double(I{Xsel(i),Csel(1)});
    X2(:,:,:,i) = im2double(I{Xsel(i),Csel(2)});
end

X = dlarray(cat(4,X1,X2),"SSCB");

end

function X = preprocessTestData(I,inputSize)
numBatch = size(I,1);
numClass = size(I,2);

X = zeros([inputSize numBatch*numClass]);

for i = 1:numClass
    for j = 1:numBatch
        X(:,:,:,j+(i-1)*numBatch) = im2double(I{j,i});
    end
end

X = dlarray(X,"SSCB");
end
