function trainKernelEstimator()
    rng(0);

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

    % Define Learn to Deblur Network
    inputSize = [128 128 1];

    lgraph = layerGraph;

    layersUnet = [
        imageInputLayer(inputSize,'Normalization','none','Name','input')

        groupedConvolution2dLayer(5,1,'channel-wise') % 124 x 124
        %batchNormalizationLayer()
        convolution2dLayer(1,16)
        %batchNormalizationLayer()
        %dropoutLayer(0.1)
        leakyReluLayer('Name','feat_3')
        maxPooling2dLayer(2,'Stride',[2 2]) % 62 x 62

        groupedConvolution2dLayer(3,1,'channel-wise') % 60 x 60
        %batchNormalizationLayer()
        convolution2dLayer(1,32)
        %batchNormalizationLayer()
        %dropoutLayer(0.1)
        leakyReluLayer('Name','feat_2')
        maxPooling2dLayer(2,'Stride',[2 2]) % 30 x 30

        groupedConvolution2dLayer(3,1,'channel-wise') % 28 x 28
        %batchNormalizationLayer()
        convolution2dLayer(1,64)
        %batchNormalizationLayer()
        dropoutLayer(0.1)
        leakyReluLayer('Name','feat_1')
        maxPooling2dLayer(2,'Stride',[2 2]) % 14 x 14

        transposedConv2dLayer(2,32,'Stride',[2 2]) % 28 x 28
        %batchNormalizationLayer()
        %dropoutLayer(0.1)
        leakyReluLayer()
        concatenationLayer(3,2,'Name','concat_1')
        transposedConv2dLayer(3,32) % 30 x 30
        %batchNormalizationLayer()
        %dropoutLayer(0.1)
        leakyReluLayer()

        transposedConv2dLayer(2,16,'Stride',[2 2]) % 60 x 60
        %batchNormalizationLayer()
        %dropoutLayer(0.1)
        leakyReluLayer()
        concatenationLayer(3,2,'Name','concat_2')
        transposedConv2dLayer(3,16) % 62 x 62
        %batchNormalizationLayer()
        %dropoutLayer(0.1)
        leakyReluLayer()

        transposedConv2dLayer(2,8,'Stride',[2 2]) % 124 x 124
        %batchNormalizationLayer()
        %dropoutLayer(0.1)
        leakyReluLayer()
        concatenationLayer(3,2,'Name','concat_3')
        transposedConv2dLayer(5,1) % 128 x 128
        %batchNormalizationLayer()
        %dropoutLayer(0.1)
        leakyReluLayer('Name','gradient_ref')
    ];
    
    layersEstimator = [
        windowingLayer('Name','windowing_1')
        functionLayer(@estimateLatent,'Name','latent_estimator','NumInputs',2,'Acceleratable',true)
        windowingLayer('Name','windowing_2')
        functionLayer(@estimateKernel,'Name','kernel_estimator','Acceleratable',true)
    ];

    lgraph = addLayers(lgraph,layersUnet);
    lgraph = addLayers(lgraph,layersEstimator);
    lgraph = addLayers(lgraph,windowingLayer('Name','windowing_3'));
    lgraph = connectLayers(lgraph,'feat_1','concat_1/in2');
    lgraph = connectLayers(lgraph,'feat_2','concat_2/in2');
    lgraph = connectLayers(lgraph,'feat_3','concat_3/in2');
    lgraph = connectLayers(lgraph,'input','windowing_1');
    lgraph = connectLayers(lgraph,'gradient_ref','latent_estimator/in2');
    lgraph = connectLayers(lgraph,'input','windowing_3');
    lgraph = connectLayers(lgraph,'windowing_3','kernel_estimator/in2');

    net = dlnetwork(lgraph);

%     load('l2b.mat','netl2b');
%     net = netl2b;

    % Specify Training Options
    numEpochs = 100;
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
    trainlossl2b = animatedline('Color',C(2,:));
    validlossl2b = animatedline('Color',C(4,:));
    xlabel("Iteration")
    ylabel("Loss")
    legend(["Train","Valid"])
    grid on

    % Train the network.
    iteration = 0;
    start = tic;
    lossl2b = Inf;
    
    % Loop over epochs.
    for epoch = 1:numEpochs

        % Reset and shuffle datastore.
        shuffle(mbqTrain);
        shuffle(mbqValid);
    
        % Loop over mini-batches.
        while hasdata(mbqTrain)
            iteration = iteration + 1;
    
            % Read mini-batch of data.
            [I,K] = next(mbqTrain);
    
            % Evaluate the gradients of the loss with respect to the learnable
            % parameters, and the network state  using dlfeval and the modelLoss function.
            [trainLoss,gradients,state] = dlfeval(@modelLoss,net,I,K);
            net.State = state;

            % Update the network parameters using the ADAM optimizer.
            [net,averageGrad,averageSqGrad] = adamupdate(net, ...
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
            addpoints(trainlossl2b,iteration,trainLoss)
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
    
            % Every validationFrequency iterations, evaluate validatio loss
            if mod(iteration,validationFrequency) == 0 || iteration == 1

                [I,K] = next(mbqValid);

                validLoss = dlfeval(@modelLoss,net,I,K);

                if validLoss < lossl2b
                    lossl2b = validLoss;
                    netl2b = net;
                end
    
                % Display the vaidation progress.
                D = duration(0,0,toc(start),'Format',"hh:mm:ss");
                validLoss = double(validLoss);
                addpoints(validlossl2b,iteration,validLoss)
                title("Epoch: " + epoch + ", Elapsed: " + string(D))
                drawnow
            end
        end
    end

    save('l2b.mat','netl2b','lossl2b');
end

function X = estimateLatent(Y,R)
    % Calculate the FFT of the inputs
    Fy = fft(fft(real(Y),128,1),128,2);
    Fr = fft(fft(real(R),128,1),128,2);
    Fk = ones(128);

    % Calculate the FFT of the priors
    G1 = fft(fft([1 -1],128,1),128,2);
    G2 = fft(fft([1 -1]',128,1),128,2);

    % Precalculate the priors
    P = 1e-3 * (abs(G1) .^ 2 + abs(G2) .^ 2);

    % Estimate the latent image
    num = conj(Fk) .* Fy + P .* Fr;
    den = abs(Fk) .^ 2 + P;
    X = real(otf2psf2(num ./ den));
end

function K = estimateKernel(X,Y)
    % Calculate the FFT of the inputs
    Fy = fft(fft(real(Y),128,1),128,2);
    Fx = fft(fft(real(X),128,1),128,2);

    % Estimate the latent image
    num = conj(Fy) .* Fx;
    den = abs(Fx) .^ 2 + 1e-4;
    K = real(otf2psf2(num ./ den));
end

function [loss,gradients,state] = modelLoss(net,I,K)
    % Calculate the predictions.
    [Ke,state] = forward(net,I);

    
%     % Normalise the kernels
%     Ke = reshape(Ke, [prod(size(Ke,[1 2])) size(Ke,3)*size(Ke,4)]);
%     Ke = Ke ./ sqrt(sum(Ke.^2,1));
%     K = reshape(K, size(Ke));
%     K = K ./ sqrt(sum(K.^2,1));

    Ke = reshape(Ke,prod(size(Ke,[1 2])),[]);
    K = reshape(K,size(Ke));
    E = K - Ke;


    % Calculate the dot product.
%     loss = mean((1 - sum(K .* Ke,1)) .^ 2);
    loss = mean(sum(E .^ 2,1));
    
    % Calculate gradients of loss with respect to learnable parameters.
    gradients = dlgradient(loss,net.Learnables);
end

function ds = prepareDataset(ds)

    % Get the kernels
    ks = transform(ds, @getKernel, 'IncludeInfo', true);
    
    % Combine the datastores
    ds = combine(ds,ks);
end

function [I,K] = preprocessMiniBatch(I,K)

    % Convert images to double
    for i = 1:length(I)
        I{i} = im2single(I{i});
    end

    % Concatenate images
    I = dlarray(cat(4,I{:}),'SSCB');

    % Concatenate kernels
    K = dlarray(cat(4,K{:}),'SSCB');
end

function [kern,info] = getKernel(~,info)

    % Get the kernel info
    sizeKern = [128 128];
    kn = double(info.Label);

    % Load the kernels
    if contains(info.Filename,'train')
        load('data/kern/train.mat','kern');
        %kern = im2double(imread(sprintf('data/kern/train/%03d.png',kern)));
    else
        load('data/kern/valid.mat','kern');
        %kern = im2double(imread(sprintf('data/kern/valid/%03d.png',kern)));
    end

    kern = kern{kn};

    % Recentre the kernels
    kern = otf2psf(psf2otf(kern,sizeKern),sizeKern);
    %kern = kern / sqrt(kern(:)' * kern(:));
end