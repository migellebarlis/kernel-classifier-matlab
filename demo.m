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
%     batchNormalizationLayer(Name='norm')
%     fft2Layer(Name='fft2')];
%     fullyConnectedLayer(numClasses, Name='full')
%     softmaxLayer(Name='softmax')];

% Create a dlnetwork object from the layer array.
net = dlnetwork(layers)

% Train for 300 epochs
numEpochs = 300;
batchSize = 20;

% Specify the options for SGDM optimization. Specify an initial learn rate
% of 0.01 with a decay of 0.01, and momentum 0.9.
initialLearnRate = 0.01;
decay = 0.01;
momentum = 0.9;

% Initialize the training progress plot.
figure
C = colororder;
lineLossTrain = animatedline(Color=C(2,:));
ylim([0 inf])
xlabel("Epoch")
ylabel("Loss")
grid on

% Initialize the velocity parameter for the SGDM solver.
velocity = [];

% Train the network using a custom training loop.
start = tic;

% Loop over epochs.
for epoch = 1:numEpochs
    X1 = zeros([inputSize batchSize]);
    X2 = zeros([inputSize batchSize]);
    Xsel = randsample(numImgs/numClasses,batchSize);

    for i = 1:batchSize
        Csel = randsample(numClasses,2);
        X1(:,:,:,i) = im2double(ims{Xsel(i),Csel(1)});
        X2(:,:,:,i) = im2double(ims{Xsel(i),Csel(2)});
    end

    % Evaluate the model gradients, state, and loss using dlfeval and the
    % modelLoss function and update the network state.
    [loss,gradients,state] = dlfeval(@modelLoss,net,X1,X2);
    net.State = state;
    
    % Determine learning rate for time-based decay learning rate schedule.
    learnRate = initialLearnRate/(1 + decay*epoch);
    
    % Update the network parameters using the SGDM optimizer.
    [net,velocity] = sgdmupdate(net,gradients,velocity,learnRate,momentum);
    
    % Display the training progress.
    D = duration(0,0,toc(start),Format="hh:mm:ss");
    loss = double(loss);
    addpoints(lineLossTrain,epoch,loss)
    title("Epoch: " + epoch + ", Elapsed: " + string(D))
    drawnow
end

function [loss,gradients,state] = modelLoss(net,X1,X2)
% Concatenate model inputs
X = dlarray(cat(4,X1,X2),"SSCB");

% Forward data through network.
[Y,state] = forward(net,X);

numFeat = numel(Y(:,:,:,1));
numBatch = size(Y,4)/2;

Y = reshape(Y,[numFeat numBatch*2]);
loss = dlarray(zeros([numBatch 1]));

% Calculate normalised dot product.
for i = 1:numBatch
    loss(i) = Y(:,i)' * Y(:,numBatch+i) ./ sqrt(Y(:,i)' * Y(:,i)) ./ sqrt(Y(:,numBatch+i)' * Y(:,numBatch+i));
end

loss = sum(loss.*loss);

% Calculate gradients of loss with respect to learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end
