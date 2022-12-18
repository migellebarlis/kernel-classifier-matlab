function [x,y] = randBlurKernel(steps)
    if nargin == 0
        steps = 100;
    end

    % Generate a covariance matrix for the given size
    x = linspace(0,1,steps);
    d = abs(x - x');
    sigma = 0.25;
    l = 0.3;
    C = (sigma .^ 2) * (1 + sqrt(5) * d / l + 5 * (d .^ 2) / (3 * l ^ 2)) .* exp(-sqrt(5) * d / l);
    
    % Generate the x and y coordinates of the path using the covariance
    % matrix
    x = mvnrnd(zeros(steps,1),C)';
    y = mvnrnd(zeros(steps,1),C)';
end