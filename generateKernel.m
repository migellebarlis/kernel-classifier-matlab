clear; clc; close all;
for i = 1:50
    [x,y] = randBlurKernel(randsample(100:20:1000,1));
    [k] = sub2im(x,y);
    subplot(5,10,i)
    imshow(k)
end