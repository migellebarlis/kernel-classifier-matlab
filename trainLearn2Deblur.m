function trainLearn2Deblur()
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

    layers = [
        imageInputLayer(inputSize,Normalization='none',Name='input_l2d')
        convolution2dLayer(5,8,Name='conv_l2d',WeightsInitializer='ones')
        tanhLayer(Name='tanh1_l2d')
        convolution2dLayer(1,8,Name='lin1_l2d')
        %tanhLayer(Name='tanh2_l2d')
        convolution2dLayer(1,4,Name='lin2_l2d')
        % quotientLayer(Name='quot_l2d')
        functionLayer(@quotientFunc,Formattable=true,Name='quot_l2d')
    ];
    
    net = dlnetwork(layers);

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

function Z = quotientFunc(X)
    Beta_k = 1e-4;
    splitChannelNum = size(X,finddim(X,'C'))/2;

    W = barthannwin(124);
    W = W * W';

    x = real(stripdims(X(:,:,1:splitChannelNum,:))) .* W;
    y = real(stripdims(X(:,:,splitChannelNum+1:end,:))) .* W;

    Fx = fft(fft(x,[],1),[],2);
    Fy = fft(fft(y,[],1),[],2);
%     F = dftmtx(size(x,1));
%     Fx = pagemtimes(pagemtimes(F',x),F);
%     Fy = pagemtimes(pagemtimes(F',y),F);
    
    num = sum(conj(Fx).*Fy,3);
    den = sum(abs(Fx).^2,3) + Beta_k;

    Z = otf2psf2(num ./ den);
%    Z = abs(pagemtimes(F,pagemtimes(num ./ den,F')));
    %Z = real(ifft(ifft(num ./ den,[],2),[],1));
    %Z = real(otf2psf2(Y_hat./X_hat));
    %Z = dlarray(Z,'SSCB');
end

function [loss,gradients,state] = modelLoss(net,I,K)
    % Calculate the predictions.
    [Ke,state] = forward(net,I);

    
    % Normalise the kernels
    Ke = reshape(Ke, [prod(size(Ke,[1 2])) size(Ke,3)*size(Ke,4)]);
    Ke = Ke ./ sqrt(sum(Ke.^2,1));
    K = reshape(K, size(Ke));
    K = K ./ sqrt(sum(K.^2,1));

%     Ke = reshape(Ke,prod(size(Ke,[1 2])),[]);
%     K = reshape(K,size(Ke));
%     E = K - Ke;


    % Calculate the dot product.
    loss = mean((1 - sum(K .* Ke,1)) .^ 2);
%     loss = mean(sum(E .^ 2,1));
    
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
    sizeKern = [124 124];
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