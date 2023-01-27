function trainBlurKern()
    % Load Training Data
    trainDS = imageDatastore('data/train', ...
        IncludeSubfolders=true, ...
        LabelSource='foldernames', ...
        FileExtensions='.png');

    trainDS = prepareDataset(trainDS);

    % Load Validation Data
    validDS = imageDatastore('data/valid', ...
        IncludeSubfolders=true, ...
        LabelSource='foldernames', ...
        FileExtensions='.png');

    validDS = prepareDataset(validDS);

    % Define Generator Network
    numFilters = 64;

    inputSize = [6 6 4];

    layers = [
        imageInputLayer(inputSize, 'Normalization', 'none', 'Name', 'input')
        transposedConv2dLayer(3,8*numFilters)
        batchNormalizationLayer
        reluLayer
        transposedConv2dLayer(3,4*numFilters,Stride=2,Cropping="same")
        batchNormalizationLayer
        reluLayer
        transposedConv2dLayer(3,2*numFilters,Stride=2,Cropping="same")
        batchNormalizationLayer
        reluLayer
        transposedConv2dLayer(5,numFilters,Stride=2,Cropping="same")
        batchNormalizationLayer
        reluLayer
        transposedConv2dLayer(5,1,Stride=2,Cropping="same")
        tanhLayer
    ];
    
    netG = dlnetwork(layers);

    % Specify Training Options
    numEpochs = 20;
    miniBatchSize = 160;

    % Specify the options for Adam optimization
    learnRate = 0.0002;
    gradientDecayFactor = 0.5;
    squaredGradientDecayFactor = 0.999;
    
    % Display the generated validation images every 25 iterations.
    validationFrequency = 25;

    % Use minibatchqueue to process and manage the mini-batches of images.
    mbqTrain = minibatchqueue(trainDS, ...
        MiniBatchSize=miniBatchSize, ...
        MiniBatchFcn=@preprocessMiniBatch, ...
        MiniBatchFormat='SSCB');

    mbqValid = minibatchqueue(validDS, ...
        MiniBatchSize=miniBatchSize, ...
        MiniBatchFcn=@preprocessMiniBatch, ...
        MiniBatchFormat='SSCB');

    % Initialize the parameters for Adam optimization.
    averageGrad = [];
    averageSqGrad = [];

    % Initialize the training progress plot.
    close all;
    figure
    C = colororder;
    lineLossTrain = animatedline('Color',C(2,:));
    lineLossValid = animatedline('Color',C(4,:));
    xlabel("Iteration")
    ylabel("Loss")
    legend(["Train","Valid"])
    grid on

    % Train the network.
    iteration = 0;
    start = tic;
    bestValG = Inf;
    
    % Loop over epochs.
    for epoch = 1:numEpochs

        % Reset and shuffle datastore.
        shuffle(mbqTrain);
        shuffle(mbqValid);
    
        % Loop over mini-batches.
        while hasdata(mbqTrain)
            iteration = iteration + 1;
    
            % Read mini-batch of data.
            [F,K] = next(mbqTrain);
    
            % Evaluate the gradients of the loss with respect to the learnable
            % parameters, and the network state  using dlfeval and the modelLoss function.
            [trainLoss,gradients,state] = dlfeval(@modelLoss,netG,F,K);
            netG.State = state;

            % Update the network parameters using the ADAM optimizer.
            [netG,averageGrad,averageSqGrad] = adamupdate(netG, ...
                gradients, ...
                averageGrad, ...
                averageSqGrad, ...
                iteration, ...
                learnRate, ...
                gradientDecayFactor, ...
                squaredGradientDecayFactor);

            % Display the training progress.
            D = duration(0,0,toc(start),'Format',"hh:mm:ss");
            trainLoss = double(trainLoss);
            addpoints(lineLossTrain,iteration,trainLoss)
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
    
            % Every validationFrequency iterations, evaluate validatio loss
            if mod(iteration,validationFrequency) == 0 || iteration == 1

                [FValid,KValid] = next(mbqValid);

                validLoss = dlfeval(@modelLoss,netG,FValid,KValid);

                if validLoss < bestValG
                    bestValG = validLoss;
                    bestNetG = netG;
                end
    
                % Display the vaidation progress.
                D = duration(0,0,toc(start),'Format',"hh:mm:ss");
                validLoss = double(validLoss);
                addpoints(lineLossValid,iteration,validLoss)
                title("Epoch: " + epoch + ", Elapsed: " + string(D))
                drawnow
            end
        end
    end

    save('bestNetG.mat','bestNetG','bestValG');
end

function [loss,gradients,state] = modelLoss(net,F,K)

    % Calculate the predictions.
    [KGenerated,state] = forward(net,F);
    
    % Calculate the l2loss.
    loss = l2loss(KGenerated,K);
    
    % Calculate gradients of loss with respect to learnable parameters.
    gradients = dlgradient(loss,net.Learnables);
end

function ds = prepareDataset(ds)

    % Get the kernels
    ks = transform(ds, @getKernel, 'IncludeInfo', true);
    
    % Combine the datastores
    ds = combine(ds,ks);
end

function [F,K] = preprocessMiniBatch(I,K)

    % Load the feature extraction model
    load('bestNet.mat','bestNet')

    for i = 1:length(I)
        I{i} = double(I{i});
    end

    % Concatenate images
    I = dlarray(cat(4,I{:}),'SSCB');

    % Get the feature
    F = predict(bestNet,I);

    K = dlarray(cat(4,K{:}),'SSCB');
end

function [kern,info] = getKernel(~,info)

    % Get the kernel info
    sizeKern = [128 128];
    kern = double(info.Label);

    % Load the kernels
    if contains(info.Filename,'train')
        kern = im2double(imread(sprintf('data/kern/train/%03d.png',kern)));
    else
        kern = im2double(imread(sprintf('data/kern/valid/%03d.png',kern)));
    end

    % Recentre the kernels
    kern = otf2psf(psf2otf(kern,sizeKern),sizeKern);
    kern = kern / sqrt(kern(:)' * kern(:));
end