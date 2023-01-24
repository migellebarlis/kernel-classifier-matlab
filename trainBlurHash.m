function trainBlurHash()
    % Reset the random number generatore
    rng(0);

    % Initialise the training image datastore
    trainDS = imageDatastore('data/train', ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames', ...
        'FileExtensions', '.png');

    % Create a shuffled copy of the training set for the pairwise training
    trainDS = prepareDataset(trainDS);

    % Initialise the validation and test image datastore
    validDS = imageDatastore('data/valid', ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames', ...
        'FileExtensions', '.png');
    [validDS,testDS] = validDS.splitEachLabel(0.5);

    % Create shuffled copies of the testing and validation sets
    validDS = prepareDataset(validDS);
    testDS = prepareDataset(testDS);
    
    % The network used in this example requires input images of size
    % 128-by-128.
    inputSize = [128 128 1];
    
    % Define the network for image classification.
    layers = [
        imageInputLayer(inputSize, 'Normalization', 'none', 'Name', 'input')
        
        % Stage 1 (128 x 128 > 62 x 62)
        groupedConvolution2dLayer(5,1,'channel-wise')
        batchNormalizationLayer()
        convolution2dLayer(1,16,'Name','feat_1') % 124 x 124
        batchNormalizationLayer()
        dropoutLayer(0.1)
        leakyReluLayer()
        maxPooling2dLayer(2,'Stride',[2 2]) % 62 x 62

        % Stage 2 (62 x 62 > 30 x 30)
        groupedConvolution2dLayer(3,1,'channel-wise')
        batchNormalizationLayer()
        convolution2dLayer(1,8,'Name','feat_2') % 60 x 60
        batchNormalizationLayer()
        dropoutLayer(0.1)
        leakyReluLayer()
        maxPooling2dLayer(2,'Stride',[2 2]) % 30 x 30

        % Stage 3 (30 x 30 > 14 x 14)
        groupedConvolution2dLayer(3,1,'channel-wise')
        batchNormalizationLayer()
        convolution2dLayer(1,4, 'Name','feat_3') % 28 x 28
        batchNormalizationLayer()
        dropoutLayer(0.1)
        leakyReluLayer()
        maxPooling2dLayer(2,'Stride',[2 2]) % 14 x 14

        % Stage 4 (14 x 14 > 6 x 6)
        groupedConvolution2dLayer(3,1,'channel-wise')
        batchNormalizationLayer()
        convolution2dLayer(1,4, 'Name', 'feat_4') % 12 x 12
        batchNormalizationLayer()
        dropoutLayer(0.1)
        leakyReluLayer()
        maxPooling2dLayer(2,'Stride',[2 2]) % 6 x 6

        % Flatten 
%        flattenLayer('Name', 'flatten')
%         fullyConnectedLayer(144)
%         tanhLayer()
%         dropoutLayer(0.3)
%         fullyConnectedLayer(144)
%         fullyConnectedLayer(500,'Name','dense')
%         tanhLayer()
%         fullyConnectedLayer(200,'Name','dense')
        ];
%     layers = [
%         imageInputLayer(inputSize, 'Normalization', 'none', 'Name', 'input')
%         
%         % Stage 1 (128 x 128 > 62 x 62)
%         convolution2dLayer(5,16, 'Name', 'feat_1') % 124 x 124
%         batchNormalizationLayer()
%         leakyReluLayer('Name', 'relu_1')
%         maxPooling2dLayer(2,'Stride', [2 2], 'Name', 'pool_1') % 62 x 62
% 
%         % Stage 2 (62 x 62 > 30 x 30)
%         convolution2dLayer(3,8, 'Name', 'feat_2') % 60 x 60
%         batchNormalizationLayer()
%         leakyReluLayer('Name', 'relu_2')
%         maxPooling2dLayer(2,'Stride', [2 2], 'Name', 'pool_2') % 30 x 30
% 
%         % Stage 3 (30 x 30 > 14 x 14)
%         convolution2dLayer(3,4, 'Name', 'feat_3') % 28 x 28
%         batchNormalizationLayer()
%         leakyReluLayer('Name', 'relu_3')
%         maxPooling2dLayer(2,'Stride', [2 2],'Name', 'pool_3') % 14 x 14
% 
%         % Stage 4 (14 x 14 > 6 x 6)
%         convolution2dLayer(3,4, 'Name', 'feat_4') % 12 x 12
%         batchNormalizationLayer()
%         leakyReluLayer('Name', 'relu_4')
%         maxPooling2dLayer(2,'Stride', [2 2],'Name', 'pool_4') % 6 x 6
% 
%         % Flatten 
%         flattenLayer('Name', 'flatten') % 
% %         fullyConnectedLayer(500,'Name','dense')
% %         tanhLayer()
% %         fullyConnectedLayer(200,'Name','dense')
%         ];
    
    % Create a dlnetwork object from the layer array.
    net = dlnetwork(layers);
    
    % Train for n epochs
    numEpochs = 20;

    mbqTrain = minibatchqueue(trainDS, ...
        'MiniBatchSize',160, ...
        'MiniBatchFormat',{'SSBC','SSBC','SSBC','SSBC','','','',''}, ...
        'OutputAsDlarray',[ones(1,4) zeros(1,4)], ...
        'OutputEnvironment',{'auto','auto','auto','auto','auto','auto','auto','auto'});
        %'OutputEnvironment',{'cpu','cpu','cpu','cpu','cpu','cpu','cpu','cpu'});

    mbqValid = minibatchqueue(validDS, ...
        'MiniBatchSize',160, ...
        'MiniBatchFormat',{'SSBC','SSBC','SSBC','SSBC','','','',''}, ...
        'OutputAsDlarray',[ones(1,4) zeros(1,4)], ...
        'OutputEnvironment',{'auto','auto','auto','auto','auto','auto','auto','auto'});
    
    % Initialize the training progress plot.
    close all;
    figure
    C = colororder;
    %yyaxis left;
    lineLossTrain = animatedline('Color',C(2,:));
    %yyaxis right;
    lineLossValid = animatedline('Color',C(4,:));
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
            [X1,X2,X3,X4,k1,k2,k3,k4] = next(mbqTrain);
            
            % Rearrange the data
            X = cat(4,repmat(X1,[1 1 1 3]),X2,X3,X4);
            KtK = repmat(k1,[1 3])' * [k2 k3 k4];
    
            % Evaluate the model gradients, state, and loss using dlfeval and the
            % modelLoss function and update the network state.
            [trainLoss,gradients,state] = dlfeval(@modelLoss,net,X,KtK);
            net.State = state;
    
            % Update the network parameters using the ADAM optimizer.
            [net,averageGrad,averageSqGrad] = adamupdate(net,gradients,averageGrad,averageSqGrad,iter);

            % Display the training progress.
            D = duration(0,0,toc(start),'Format',"hh:mm:ss");
            trainLoss = double(trainLoss);
            addpoints(lineLossTrain,iter,trainLoss)
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow

            if mod(iter,25) == 0
                % Get the next mini-batch
                [X1,X2,X3,X4,k1,k2,k3,k4] = next(mbqValid);
                
                % Rearrange the data
                X = cat(4,repmat(X1,[1 1 1 3]),X2,X3,X4);
                KtK = repmat(k1,[1 3])' * [k2 k3 k4];

                loss = dlfeval(@modelLoss,net,X,KtK);

                if loss < bestVal
                    bestVal = loss;
                    bestNet = net;
                end
                
                % Display the training progress.
                D = duration(0,0,toc(start),'Format',"hh:mm:ss");
                loss = double(loss);
                addpoints(lineLossValid,iter,loss)
                title("Epoch: " + epoch + ", Elapsed: " + string(D))
                drawnow
            end

            iter = iter + 1;            
        end
    end
    
    save('bestNet.mat','bestNet','bestVal');
end

function ds = prepareDataset(ds)
    % Get the kernel and image numbers
    kern = double(ds.Labels);%
    im = cellfun(@(x)(str2double(x(end-7:end-4))),ds.Files);
    
    % Create a fully scrambled dataset
    ind = randperm(length(kern));
    mask = im == im(ind) | kern == kern(ind);
    while any(mask)
        % Scramble the affected points
        n = nnz(mask);
        t = ind(mask);
        ind(mask) = t(randperm(n));
        
        % Check for any remaining matches
        mask = im == im(ind) | kern == kern(ind);
    end
    ds2 = ds.subset(ind);
    
    % Create a kernel-scrambled dataset
    ind = 1:numel(ds.Files);
    ims = unique(im);
    for i = 1:numel(ims)
        % Find all relevant files
        mask = im == ims(i);
        
        % Scramble the order for the current image
        while any(im == ims(i) & kern == kern(ind))
            t = ind(mask);
            ind(mask) = t(randperm(numel(t)));
        end
    end
    ds3 = ds.subset(ind);
    
    % Create a kernel-scrambled dataset
    ind = 1:numel(ds.Files);
    kerns = unique(kern);
    for i = 1:numel(kerns)
        % Find all relevant files
        mask = kern == kerns(i);
        
        % Scramble the order for the current image
        while any(kern == kerns(i) & im == im(ind))
            t = ind(mask);
            ind(mask) = t(randperm(numel(t)));
        end
    end
    ds4 = ds.subset(ind);
    
    k1 = transform(ds, @getKernel, 'IncludeInfo', true);
    k2 = transform(ds2, @getKernel, 'IncludeInfo', true);
    k3 = transform(ds3, @getKernel, 'IncludeInfo', true);
    k4 = transform(ds4, @getKernel, 'IncludeInfo', true);
    
    % Combine the datastores
    ds = combine(ds,ds2,ds3,ds4,k1,k2,k3,k4);
end

function [kern,info] = getKernel(~,info)
    % Get the kernel info
    kern = double(info.Label);

    % Load the kernels
    if contains(info.Filename,'train')
        kern = im2double(imread(sprintf('data/kern/train/%03d.png',kern)));
    else
        kern = im2double(imread(sprintf('data/kern/valid/%03d.png',kern)));
    end

    % Recentre the kernels
    kern = otf2psf(psf2otf(kern,[150 150]),[150 150]);
    kern = kern(:);
    kern = kern / sqrt(kern' * kern);
end

function [loss,gradients,state] = modelLoss(net,X,KtK)
    % Process the data through network.
    [T,state] = forward(net,X);
    T = reshape(T,prod(size(T,1:3)),[],2);

    % Split the features from the two images
    X = T(:,:,1);
    Y = T(:,:,2);

    % Create penalties for feature vectors approaching zero
    lambda = 1;
    p1 = lambda * exp(-(sum(X .* X) .^ 2) ./ 0.01);
    p2 = lambda * exp(-(sum(Y .* Y) .^ 2) ./ 0.01);
    p = mean((p1 + p2) / 2);

    % Normalise the features
    X = X ./ sqrt(sum(X .^ 2));
    Y = Y ./ sqrt(sum(Y .^ 2));
    
    % Calculate the main loss
    loss = mean((KtK(:) - reshape(X' * Y,[],1)) .^ 2) + p;
    %loss = sum(featSim .^ 2) + p;

    % Calculate gradients of loss with respect to learnable parameters.
    gradients = dlgradient(loss,net.Learnables);
end
