function trainBlurHash()
    % Reset the random number generatore
    rng(0);

    % Initialise the training image datastore
    trainDS = imageDatastore('data/train', ...
        IncludeSubfolders=true, ...
        LabelSource='foldernames', ...
        FileExtensions='.png');

    % Create a shuffled copy of the training set for the pairwise training
    trainShuffled = trainDS.shuffle();

    % Calculate the kernel similarities and create a combine datastore
    kernSim = transform(trainDS,trainShuffled,@kernelSimilarity,IncludeInfo=true);
    trainDS = combine(trainDS,trainShuffled,kernSim);

    % Initialise the validation and test image datastore
    validDS = imageDatastore('data/valid', ...
        IncludeSubfolders=true, ...
        LabelSource='foldernames', ...
        FileExtensions='.png');
    [validDS,testDS] = validDS.splitEachLabel(0.5);

    % Process the datastores
    validShuffled = validDS.shuffle();
    kernSim = transform(validDS,validShuffled,@kernelSimilarity,IncludeInfo=true);
    validDS = combine(validDS,validShuffled,kernSim);

    testShuffled = testDS.shuffle();
    kernSim = transform(testDS,testShuffled,@kernelSimilarity,IncludeInfo=true);
    testDS = combine(testDS,testShuffled,kernSim);
    
    % The network used in this example requires input images of size
    % 128-by-128.
    inputSize = [128 128 1];
    
    % Define the network for image classification.
    layers = [
        imageInputLayer(inputSize, Normalization='none', Name='input')
        
        % Stage 1 (128 x 128 > 62 x 62)
        convolution2dLayer(5,16, Name='feat_1')
        batchNormalizationLayer()
        leakyReluLayer(Name='relu_1')
        maxPooling2dLayer(2,Stride=[2 2],Name='pool_1')

        % Stage 2 (62 x 62 > 30 x 30)
        convolution2dLayer(3,8, Name='feat_2')
        batchNormalizationLayer()
        leakyReluLayer(Name='relu_2')
        maxPooling2dLayer(2,Stride=[2 2],Name='pool_2')

        % Stage 3 (30 x 30 > 14 x 14)
        convolution2dLayer(3,4, Name='feat_3')
        batchNormalizationLayer()
        leakyReluLayer(Name='relu_3')
        maxPooling2dLayer(2,Stride=[2 2],Name='pool_3')

        % Flatten
        flattenLayer(Name='flatten') % 
        ];
    
    % Create a dlnetwork object from the layer array.
    net = dlnetwork(layers);
    
    % Train for n epochs
    numEpochs = 100;

    mbqTrain = minibatchqueue(trainDS, ...
        MiniBatchSize=160, ...
        MiniBatchFormat={'SSBC','SSBC',''}, ...
        OutputAsDlarray=[1 1 0], ...
        OutputEnvironment={'gpu','gpu','gpu'});

    mbqValid = minibatchqueue(trainDS, ...
        MiniBatchSize=160, ...
        MiniBatchFormat={'SSBC','SSBC',''}, ...
        OutputAsDlarray=[1 1 0], ...
        OutputEnvironment={'gpu','gpu','gpu'});
    
    % Initialize the training progress plot.
    close all;
    figure
    C = colororder;
    %yyaxis left;
    lineLossTrain = animatedline(Color=C(2,:));
    %yyaxis right;
    lineLossValid = animatedline(Color=C(4,:));
    xlabel("Iteration")
    ylabel("Loss")
    legend(["Train","Valid"])
    grid on
    
    % Initialize the gradient parameter for the ADAM solver.
    averageGrad = [];
    averageSqGrad = [];
    
    % Loop over epochs.
    iter = 1;
    bestVal = Inf;
    start = tic;
    for epoch = 1:numEpochs
        % Shuffle the data
        shuffle(mbqTrain);
        shuffle(mbqValid);

        while hasdata(mbqTrain)
            % Get the next mini-batch
            [X,Y,k] = next(mbqTrain);
    
            % Evaluate the model gradients, state, and loss using dlfeval and the
            % modelLoss function and update the network state.
            [trainLoss,gradients,state] = dlfeval(@modelLoss,net,cat(4,X,Y),k);
            net.State = state;
    
            % Update the network parameters using the ADAM optimizer.
            [net,averageGrad,averageSqGrad] = adamupdate(net,gradients,averageGrad,averageSqGrad,iter);

            % Display the training progress.
            D = duration(0,0,toc(start),Format="hh:mm:ss");
            trainLoss = double(trainLoss);
            addpoints(lineLossTrain,iter,trainLoss)
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow

            if mod(iter,25) == 0
                [X,Y,k] = next(mbqValid);
                loss = dlfeval(@modelLoss,net,cat(4,X,Y),k);

                if loss < bestVal
                    bestVal = loss;
                    bestNet = net;
                end
                
                % Display the training progress.
                D = duration(0,0,toc(start),Format="hh:mm:ss");
                loss = double(loss);
                addpoints(lineLossValid,iter,loss)
                title("Epoch: " + epoch + ", Elapsed: " + string(D))
                drawnow
            end

            iter = iter + 1;            
        end
    end
    
    save('bestNet.mat','bestNet');
end

function [ds,info] = kernelSimilarity(~,~,info1,info2)
    % Get the kernel info
    k1 = double(info1.Label);
    k2 = double(info2.Label);

    % Load the kernels
    k1 = im2double(imread(sprintf('data/kern/train/%03d.png',k1)));
    k2 = im2double(imread(sprintf('data/kern/train/%03d.png',k2)));

    % Recentre the kernels
    k1 = otf2psf(psf2otf(k1,[150 150]),[150 150]);
    k2 = otf2psf(psf2otf(k2,[150 150]),[150 150]);

    % Calculate the similarity
    ds = (k1(:)' * k2(:)) ./ (sqrt((k1(:)' * k1(:)) * (k2(:)' * k2(:))) + eps);
    info = [];
end

function [loss,gradients,state] = modelLoss(net,X,k)
    % Process the data through network.
    [T,state] = forward(net,X);
    T = reshape(T,size(T,1),[],2);

    % Split the features from the two images
    X = T(:,:,1);
    Y = T(:,:,2);

    % Create penalties for feature vectors approaching zero
    lambda = 1;
    p1 = lambda * exp(-(sum(X .* X) .^ 2) ./ 0.01);
    p2 = lambda * exp(-(sum(Y .* Y) .^ 2) ./ 0.01);
    p = sum(p1 + p2);

    % Calculate normalised dot product.
    featSim = sum(X .* Y) ./ (sqrt(sum(X .* X) .* sum(Y .* Y)) + eps);
    loss = (featSim .^ 2) * (1 - k) + ((1 - featSim) .^ 2) * k + p;
    %loss = sum(featSim .^ 2) + p;

    % Calculate gradients of loss with respect to learnable parameters.
    gradients = dlgradient(loss,net.Learnables);
end

function loss = modelValidate(net,X)
% Forward data through network.
Y = predict(net,X);

if ismatrix(Y)
    numFeat = size(Y,1);
    numBatch = size(Y,2) / 4;
else
    numFeat = numel(Y(:,:,:,1));
    numBatch = size(Y,4)/4;
end

Y = reshape(Y,[numFeat numBatch 4]);
Y1 = squeeze(Y(:,:,1));
Y2 = squeeze(Y(:,:,2));
Y3 = squeeze(Y(:,:,3));
Y4 = squeeze(Y(:,:,4));

% Create penalties for feature vectors approaching zero
lambda = 1;
p1 = lambda * exp(-(sum(Y1 .* Y1) .^ 2) ./ 0.01);
p2 = lambda * exp(-(sum(Y2 .* Y2) .^ 2) ./ 0.01);
p3 = lambda * exp(-(sum(Y3 .* Y3) .^ 2) ./ 0.01);
p4 = lambda * exp(-(sum(Y4 .* Y4) .^ 2) ./ 0.01);
p = sum(p1 + p2 + p3 + p4);

% Calculate normalised dot product.
t1 = sum((sum(Y1 .* Y2) ./ (sqrt(sum(Y1 .* Y1) .* sum(Y2 .* Y2)) + eps)) .^ 2);
t2 = sum((1 - sum(Y3 .* Y4) ./ (sqrt(sum(Y3 .* Y3) .* sum(Y4 .* Y4)) + eps)) .^ 2);
loss = t1 + t2 + p;

end

function loss = modelTest(net,X)
% Make prediction.
Y = predict(net,X);

if ismatrix(Y)
    numFeat = size(Y,1);
    numBatch = size(Y,2) / 4;
else
    numFeat = numel(Y(:,:,:,1));
    numBatch = size(Y,4)/4;
end

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
X3 = zeros([inputSize numBatch]);
X4 = zeros([inputSize numBatch]);
Xsel = randperm(numBatch);

for i = 1:numBatch
    Csel = randperm(numClass,2);
    X1(:,:,:,i) = im2double(I{Xsel(i),Csel(1)});
    X2(:,:,:,i) = im2double(I{Xsel(i),Csel(2)});
end

for i = 1:numBatch
    Csel = mod(i,8) + 1;
    Xsel = randperm(numBatch,2);
    X3(:,:,:,i) = im2double(I{Xsel(1),Csel});
    X4(:,:,:,i) = im2double(I{Xsel(2),Csel});
end

X = dlarray(cat(4,X1,X2,X3,X4),"SSCB");

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
