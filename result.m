clear;clc;close all;

addpath("../libraries/export_fig");

% Load the data as an image datastore using the imageDatastore
% function and specify the folder containing the image data.
imds = imageDatastore("data", ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames", ...
    FileExtensions='.png');

% Read all images from the image datastore to a cell array
numKer = numel(categories(imds.Labels));
numImg = numel(imds.Files)/numKer;
ims = reshape(readall(imds),[numImg numKer]);

% Load the network
load('bestNet.mat');

% Preprocess for model format
X = preprocessData(ims);

% Generate the loss for all images
L = getLoss(bestNet,X,numKer);

for i = 1:size(L,3)
    imagesc(L(:,:,i))
    colorbar
    export_fig(sprintf("loss/%03d",i),"-png")
end

% Generate the features for all images
for i = 1:3
    F = double(extractdata(predict(bestNet,X,"Outputs",sprintf("feat_%d",i))));
    for j = 1:size(F,3)
        for k = 1:size(F,4)
            imshow(F(:,:,j,k),[])
            export_fig(sprintf("feature/%03d_%03d_%03d_%03d.png",mod(k-1,numImg)+1,ceil(k/numImg),i,j))
        end
    end
end

function L = getLoss(net,X,numSet)
% Make prediction.
Y = predict(net,X);

numFeat = numel(Y(:,:,:,1));
numBatch = size(Y,4)/numSet;

Y = reshape(Y,[numFeat numBatch numSet]);
L = zeros([numSet numSet numBatch]);

for i = 1:numSet
    for j = 1:numSet
        Y1 = squeeze(Y(:,:,i));
        Y2 = squeeze(Y(:,:,j));
    
        % Create penalties for feature vectors approaching zero
        lambda = 1;
        p1 = lambda * exp(-(sum(Y1 .* Y1) .^ 2) ./ 0.01);
        p2 = lambda * exp(-(sum(Y2 .* Y2) .^ 2) ./ 0.01);
        p = p1 + p2;
        
        % Calculate normalised dot product.
        L(i,j,:) = (((sum(Y1 .* Y2) .^ 2) ./ (sum(Y1 .* Y1 + Y2 .* Y2) + eps)) + p)';
    end
end

end

function X = preprocessData(I)
imgSize = size(I{1});
numImg = size(I,1);
numKer = size(I,2);

X = zeros([imgSize 1 numImg*numKer]);

for i = 1:numImg
    for j = 1:numKer
        X(:,:,:,i+(j-1)*numImg) = im2double(I{i,j});
    end
end

X = dlarray(X,"SSCB");
end