function generateImages(numKern)
    DIV2Kpath = 'G:/MATLAB/databases/DIV2K';
    if nargin == 0
        numKern = 80;
    end

    % Generate different blur kernels
    rng(0);
    kern = cell(numKern,1);
    mkdir('data/kern/train');
    for n = 1:numKern
        [x,y] = randBlurKernel(randsample(100:20:1000,1));
        kern{n} = sub2im(x,y);
        imwrite(kern{n}/max(kern{n}(:)),sprintf('data/kern/train/%03d.png',n));
    end
    save('data/kern/train.mat','kern');

    mkdir('data/train');
    
    % Create directories for each kernel
    for n = 1:numKern
        mkdir(sprintf('data/train/%03d',n));
    end

    % Create the training images
    for n = 1:800
        % Load the image
        X = im2double(imread(sprintf('%s/train/%04d.png',DIV2Kpath,n)));

        % Convert to greyscale
        X = rgb2gray(X);

        % Determine the number of kernels
        parfor i = 1:numKern
            % Apply the kernel
            T = imfilter(X,kern{i},'replicate');
    
            % Locate the busiest 128 x 128 region
            X1 = boxFilter(abs(diff(padarray(T,[1 0],'pre'),1,1)));
            X2 = boxFilter(abs(diff(padarray(T,[0 1],'pre'),1,2)));
            V = X1 + X2;
    
            [~,ind] = max(V(:));
            [r,c] = ind2sub(size(V),ind);
            imwrite(T(r:r+127,c:c+127),sprintf('data/train/%03d/%04d.png',i,n));
        end
    end
    
    % Generate different blur kernels
    rng(10);
    kern = cell(numKern,1);
    mkdir('data/kern/valid');
    for n = 1:numKern
        [x,y] = randBlurKernel(randsample(100:20:1000,1));
        kern{n} = sub2im(x,y);
        imwrite(kern{n}/max(kern{n}(:)),sprintf('data/kern/valid/%03d.png',n));
    end
    save('data/kern/valid.mat','kern');

    mkdir('data/valid');
    
    % Create directories for each kernel
    for n = 1:numKern
        mkdir(sprintf('data/valid/%03d',n));
    end

    % Create the validation images
    for n = 801:900
        % Load the image
        X = im2double(imread(sprintf('%s/valid/%04d.png',DIV2Kpath,n)));

        % Convert to greyscale
        X = rgb2gray(X);

        % Determine the number of kernels
        parfor i = 1:numKern
            % Apply the kernel
            T = imfilter(X,kern{i},'replicate');
    
            % Locate the busiest 128 x 128 region
            X1 = boxFilter(abs(diff(padarray(T,[1 0],'pre'),1,1)));
            X2 = boxFilter(abs(diff(padarray(T,[0 1],'pre'),1,2)));
            V = X1 + X2;
    
            [~,ind] = max(V(:));
            [r,c] = ind2sub(size(V),ind);
            imwrite(T(r:r+127,c:c+127),sprintf('data/valid/%03d/%04d.png',i,n));
        end
    end
end

function X = boxFilter(X)
    X = cumsum(padarray(X,[1 0],'pre'),1);
    X = X(129:end,:) - X(1:end-128,:);

    X = cumsum(padarray(X,[0 1],'pre'),2);
    X = X(:,129:end) - X(:,1:end-128);
end
