clear;clc;close all;

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

figure(1);
<<<<<<< HEAD
imagesc(L.extractdata);
grid on;
xticks(0:80:640);
yticks(0:80:640);
axis square;
% figure(1);
% for i = 1:size(L,3)
%     subplot(8,10,i);
%     imagesc(L(:,:,i))
%     %colorbar
%     %export_fig(sprintf("loss/%03d",i),"-png")
% end
=======
for i = 1:size(L,3)
    subplot(8,10,i);
    imagesc(L(:,:,i))
    %colorbar
    %export_fig(sprintf("loss/%03d",i),"-png")
end
>>>>>>> dd039bb8ae52834e096d2310b5d3e14c60c30b2a

% Generate the features for all images
mkdir('features/');
[F1,F2,F3,F4] = predict(bestNet,X,"Outputs",{'relu_1','relu_2','relu_3','pool_3'});
F{1} = reshape(F1.extractdata,[numel(F1(:,:,1,1,1)) size(F1,3) 80 8]);
F{2} = reshape(F2.extractdata,[numel(F2(:,:,1,1,1)) size(F2,3) 80 8]);
F{3} = reshape(F3.extractdata,[numel(F3(:,:,1,1,1)) size(F3,3) 80 8]);
F{4} = reshape(F4.extractdata,[numel(F4(:,:,1,1,1)) size(F4,3) 80 8]);

for i = 1:80
    for j = 1:8
        for f = 1:4
            % Determine the current size of the feature image
            n = sqrt(size(F{f},1));

            % Pad the feature image with NaNs
            mask = padarray(true(n),[round(0.1*n) round(0.1*n)]);
            T = nan([numel(mask) size(F{f},2)]);
            T(mask,:) = F{f}(:,:,i,j);
            
            % Arrange all the features in a 2:3 ratio
            % xy = m
            % y = x / 1.5
            % y^2 / 1.5 = m
            m = size(T,2);
            c = ceil(sqrt(m * 1.5));
            r = ceil(m / c);
            T = padarray(T,[0 r*c-m],nan,'post');

            % Display as an image
            n = sqrt(size(T,1));
            T = col2im(T,[n n],[r c].*[n n],'distinct');

            % Normalise the image
            T = T - min(T(:));
            T = T ./ max(T(:));
            imwrite(256*T,parula(256),sprintf('features/feat%d_im%d_kern%d.png',f,i,j));
        end
    end
end

function L = getLoss(net,X,numSet)
% Make prediction.
Y = predict(net,X);

% Calculate the inner products
Y = reshape(Y,[],size(Y,4));
num = reshape(sum(repmat(Y,[1 size(Y,2)]) .* kron(Y,ones(1,size(Y,2)))),size(Y,2),[]);
Y = sum(Y .^ 2);
den = reshape(sqrt(repmat(Y,[1 length(Y)]) .* kron(Y,ones(1,length(Y)))) + eps,length(Y),[]);
L = num ./ den;
%L = L ./ diag(L);
% numFeat = numel(Y(:,:,:,1));
% numBatch = size(Y,4)/numSet;
% 
% Y = reshape(Y,[numFeat numBatch numSet]);
% L = zeros([numSet numSet numBatch]);
% 
% for i = 1:numSet
%     for j = 1:numSet
%         Y1 = squeeze(Y(:,:,i));
%         Y2 = squeeze(Y(:,:,j));
%     
%         % Create penalties for feature vectors approaching zero
%         lambda = 1;
%         p1 = lambda * exp(-(sum(Y1 .* Y1) .^ 2) ./ 0.01);
%         p2 = lambda * exp(-(sum(Y2 .* Y2) .^ 2) ./ 0.01);
%         p = p1 + p2;
%         
%         % Calculate normalised dot product.
%         L(i,j,:) = (((sum(Y1 .* Y2) .^ 2) ./ (sum(Y1 .* Y1 + Y2 .* Y2) + eps)) + p)';
%     end
% end

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
