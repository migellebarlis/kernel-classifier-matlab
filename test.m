function test()
    % Test path
    testPath = 'test_sun';
    
    % Load the test data
    testDS = imageDatastore(strcat('data/',testPath), ...
        IncludeSubfolders=true, ...
        LabelSource='foldernames', ...
        FileExtensions='.png');

    [I,K] = prepareDataset(testDS);

    % Load the feature extraction model.
    netF = load('bestNet.mat','bestNet');
    netF = netF.bestNet;

    % Load the generator model.
    netG = load('bestNetG.mat','bestNetG');
    netG = netG.bestNetG;

    % Calculate the predictions.
    F = predict(netF,I);
    KTest = predict(netG,F);

    % Calculate the loss.
    loss = l2loss(KTest,K);

    idx = [1 81 161 241 321 401 481 561];

    figure
    for i = 1:length(idx)
        subplot(2,8,i)
        imshow(extractdata(K(:,:,:,idx(i))),[])
        subplot(2,8,length(idx)+i)
        imshow(extractdata(KTest(:,:,:,idx(i))),[])
    end
end

function [I,K] = prepareDataset(ds)

    % Get path
    path = split(ds.Folders{1},'/');
    path = path{end};

    % Initialize images.
    I = zeros([size(imread(ds.Files{1})) 1 length(ds.Files)]);

    % Initialize kernels.
    sizeK = [128 128];
    K = zeros([sizeK 1 length(ds.Files)]);

    for i = 1:length(ds.Files)

        % Load the images.
        im = im2double(imread(ds.Files{i}));
        I(:,:,1,i) = im;

        % Load the kernels.
        kern = double(ds.Labels(i));
        kern = im2double(imread(sprintf('data/kern/%s/%03d.png',path,kern)));

        % Recenter kernels.
        kern = otf2psf(psf2otf(kern,sizeK),sizeK);
        kern = kern / sqrt(kern(:)' * kern(:));
        K(:,:,1,i) = kern;
    end

    I = dlarray(I,'SSCB');
    K = dlarray(K,'SSCB');
end