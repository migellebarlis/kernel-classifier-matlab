% Load the digits data as an image datastore using the imageDatastore
% function and specify the folder containing the image data.
imds = imageDatastore("data", ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames", ...
    FileExtensions='.png');

% Partition the data into training and validation sets. Set aside 10% of
% the data for validation using the splitEachLabel function.
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.9,"randomize");

% The network used in this example requires input images of size
% 128-by-128.
inputSize = [128 128 1];

% To automatically resize the training and validation images, use an
% augmented image datastore.
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% Determine the number of classes in the training data.
classes = categories(imdsTrain.Labels);
numClasses = numel(classes);

% Define the network for image classification.
layers = [
    imageInputLayer(inputSize, Normalization='none', Name='input')
    convolution2dLayer(3, 64, Padding='same', Name='features')
%     batchNormalizationLayer(Name='norm')
    fft2Layer(Name='fft2')];
%     fullyConnectedLayer(numClasses, Name='full')
%     softmaxLayer(Name='softmax')];

% Create a dlnetwork object from the layer array.
net = dlnetwork(layers)

% Train for ten epochs with a mini-batch size of 128.
numEpochs = 10;
miniBatchSize = 128;

% Specify the options for SGDM optimization. Specify an initial learn rate
% of 0.01 with a decay of 0.01, and momentum 0.9.
initialLearnRate = 0.01;
decay = 0.01;
momentum = 0.9;

% Create a minibatchqueue object that processes and manages mini-batches of
% images during training.
mbq = minibatchqueue(augimdsTrain,...
    MiniBatchSize=miniBatchSize,...
    MiniBatchFcn=@preprocessMiniBatch,...
    MiniBatchFormat=["SSCB" ""]);

% Initialize the training progress plot.
figure
C = colororder;
lineLossTrain = animatedline(Color=C(2,:));
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on

% Initialize the velocity parameter for the SGDM solver.
velocity = [];

% Train the network using a custom training loop.
iteration = 0;
start = tic;

% Loop over epochs.
for epoch = 1:numEpochs

    % Shuffle data.
    shuffle(mbq);
    
    % Loop over mini-batches.
    while hasdata(mbq)
        iteration = iteration + 1;
        
        % Read mini-batch of data.
        [X,T] = next(mbq);
        
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelLoss function and update the network state.
        [loss,gradients,state] = dlfeval(@modelLoss,net,X,T);
        net.State = state;
        
        % Determine learning rate for time-based decay learning rate schedule.
        learnRate = initialLearnRate/(1 + decay*iteration);
        
        % Update the network parameters using the SGDM optimizer.
        [net,velocity] = sgdmupdate(net,gradients,velocity,learnRate,momentum);
        
        % Display the training progress.
        D = duration(0,0,toc(start),Format="hh:mm:ss");
        loss = double(loss);
        addpoints(lineLossTrain,iteration,loss)
        title("Epoch: " + epoch + ", Elapsed: " + string(D))
        drawnow
    end
end

% Test the classification accuracy of the model by comparing the
% predictions on the validation set with the true labels.
numOutputs = 1;

mbqTest = minibatchqueue(augimdsValidation,numOutputs, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatchPredictors, ...
    MiniBatchFormat="SSCB");

% Loop over the mini-batches and classify the images using modelPredictions
% function, listed at the end of the example.
YTest = modelPredictions(net,mbqTest,classes);

% Evaluate the classification accuracy.
TTest = imdsValidation.Labels;
accuracy = mean(TTest == YTest)

% Visualize the predictions in a confusion chart.
figure
confusionchart(TTest,YTest)

function [loss,gradients,state] = modelLoss(net,X,T)

% Forward data through network.
[Y,state] = forward(net,X);

% Calculate KL divergence loss.
P = dlarray(zeros([size(Y,1) size(Y,2) size(T,1)]));
L = dlarray(zeros([size(T,1) 1]));

t = 1:size(T,1);
normY = Y ./ sum(Y,[1 2]);

for i = t
    temp = normY(:,:,:,T(i,:)>0);
    P(:,:,i) = mean(temp,[3 4]);
end

for i = t
    for j = t(t~=i)
        temp = P(:,:,i) .* log(P(:,:,i) ./ P(:,:,j));
        temp(isnan(temp)) = 0; % resolving the case when P(i)==0
        L(i) = L(i) - sum(temp,"all");
    end
end

loss = sum(L);

% Calculate gradients of loss with respect to learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end

function Y = modelPredictions(net,mbq,classes)

Y = [];

% Loop over mini-batches.
while hasdata(mbq)
    X = next(mbq);

    % Make prediction.
    scores = predict(net,X);

    % Decode labels and append to output.
    labels = onehotdecode(scores,classes,1)';
    Y = [Y; labels];
end

end

function [X,T] = preprocessMiniBatch(dataX,dataT)

% Preprocess predictors.
X = preprocessMiniBatchPredictors(dataX);

% Extract label data from cell and concatenate.
T = cat(2,dataT{1:end});

% One-hot encode labels.
T = onehotencode(T,1);

end

function X = preprocessMiniBatchPredictors(dataX)

% Concatenate.
X = cat(4,dataX{1:end});

end
