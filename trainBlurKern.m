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

    layersGenerator = [
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
    
    netG = dlnetwork(layersGenerator);

    % Define Discriminator Network
    dropoutProb = 0.5;
    numFilters = 64;
    scale = 0.2;
    
    inputSize = [128 128 1];
    
    layersDiscriminator = [
        imageInputLayer(inputSize,Normalization="none")
        dropoutLayer(dropoutProb)
        convolution2dLayer(5,numFilters,Stride=2,Padding="same")
        leakyReluLayer(scale)
        convolution2dLayer(5,2*numFilters,Stride=2,Padding="same")
        batchNormalizationLayer
        leakyReluLayer(scale)
        convolution2dLayer(3,4*numFilters,Stride=2,Padding="same")
        batchNormalizationLayer
        leakyReluLayer(scale)
        convolution2dLayer(3,8*numFilters,Stride=2,Padding="same")
        batchNormalizationLayer
        leakyReluLayer(scale)
        convolution2dLayer(3,1)
        sigmoidLayer
    ];

    netD = dlnetwork(layersDiscriminator);

    % Specify Training Options
    numEpochs = 20;
    miniBatchSize = 160;

    % Specify the options for Adam optimization
    learnRate = 0.0002;
    gradientDecayFactor = 0.5;
    squaredGradientDecayFactor = 0.999;

    % Specify to flip the real labels with probability 0.35.
    flipProb = 0.35;
    
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
    trailingAvgG = [];
    trailingAvgSqG = [];
    trailingAvg = [];
    trailingAvgSqD = [];

    % Initialize the training progress plots. Create a figure and resize it to have twice the width.
    F = figure;
    F.Position(3) = 2*F.Position(3);
    
    % Create a subplot for the generated images and the network scores.
    imageAxes = subplot(1,2,1);
    scoreAxes = subplot(1,2,2);
    
    % Initialize the animated lines for the scores plot.
    C = colororder;
    lineScoreG = animatedline(scoreAxes,Color=C(1,:));
    lineScoreD = animatedline(scoreAxes,Color=C(2,:));
    legend("Generator","Discriminator");
    ylim([0 1])
    xlabel("Iteration")
    ylabel("Score")
    grid on

    % Train the GAN.
    iteration = 0;
    start = tic;
    
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
            % parameters, the generator state, and the network scores using
            % dlfeval and the modelLoss function.
            [~,~,gradientsG,gradientsD,stateG,scoreG,scoreD] = ...
                dlfeval(@modelLoss,netG,netD,F,K,flipProb);
            netG.State = stateG;
    
            % Update the discriminator network parameters.
            [netD,trailingAvg,trailingAvgSqD] = adamupdate(netD, gradientsD, ...
                trailingAvg, trailingAvgSqD, iteration, ...
                learnRate, gradientDecayFactor, squaredGradientDecayFactor);
    
            % Update the generator network parameters.
            [netG,trailingAvgG,trailingAvgSqG] = adamupdate(netG, gradientsG, ...
                trailingAvgG, trailingAvgSqG, iteration, ...
                learnRate, gradientDecayFactor, squaredGradientDecayFactor);
    
            % Every validationFrequency iterations, display batch of generated
            % images using the held-out generator input.
            if mod(iteration,validationFrequency) == 0 || iteration == 1
                [FValid,~] = next(mbqValid);
                % Generate images using the held-out generator input.
                KGenerated = predict(netG,FValid);
    
                % Tile and rescale the images in the range [0 1].
                I = imtile(extractdata(KGenerated));
                I = rescale(I);
    
                % Display the images.
                subplot(1,2,1);
                image(imageAxes,I)
                xticklabels([]);
                yticklabels([]);
                title("Generated Images");
            end
    
            % Update the scores plot.
            subplot(1,2,2)
            scoreG = double(extractdata(scoreG));
            addpoints(lineScoreG,iteration,scoreG);
    
            scoreD = double(extractdata(scoreD));
            addpoints(lineScoreD,iteration,scoreD);
    
            % Update the title with training progress information.
            D = duration(0,0,toc(start),Format="hh:mm:ss");
            title(...
                "Epoch: " + epoch + ", " + ...
                "Iteration: " + iteration + ", " + ...
                "Elapsed: " + string(D))
    
            drawnow
        end
    end
end

function [lossG,lossD,gradientsG,gradientsD,stateG,scoreG,scoreD] = ...
    modelLoss(netG,netD,F,K,flipProb)

    % Calculate the predictions for real data with the discriminator network.
    YReal = forward(netD,K);
    
    % Calculate the predictions for generated data with the discriminator
    % network.
    [KGenerated,stateG] = forward(netG,F);
    YGenerated = forward(netD,KGenerated);
    
    % Calculate the score of the discriminator.
    scoreD = (mean(YReal) + mean(1-YGenerated)) / 2;
    
    % Calculate the score of the generator.
    scoreG = mean(YGenerated);
    
    % Randomly flip the labels of the real images.
    numObservations = size(YReal,4);
    idx = rand(1,numObservations) < flipProb;
    YReal(:,:,:,idx) = 1 - YReal(:,:,:,idx);
    
    % Calculate the GAN loss.
    [lossG, lossD] = ganLoss(YReal,YGenerated);
    
    % For each network, calculate the gradients with respect to the loss.
    gradientsG = dlgradient(lossG,netG.Learnables,RetainData=true);
    gradientsD = dlgradient(lossD,netD.Learnables);
end

function [lossG,lossD] = ganLoss(YReal,YGenerated)
    % Calculate the loss for the discriminator network.
    lossD = -mean(log(YReal)) - mean(log(1-YGenerated));
    
    % Calculate the loss for the generator network.
    lossG = -mean(log(YGenerated));
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
        I{i} = im2double(I{i});
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