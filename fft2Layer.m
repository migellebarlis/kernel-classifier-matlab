classdef fft2Layer < nnet.layer.Layer

    methods
        function layer = fft2Layer(NameValueArgs) 
            % layer = fft2Layer(numInputs,name) creates a
            % fft2 layer and specifies the layer name.

            arguments
                NameValueArgs.Name = '';
            end

            % Set layer name.
            layer.Name = NameValueArgs.Name;
        end

        function Z = predict(layer,X)
            % Forward input data through the layer at prediction time and
            % output the result and updated state.
            %
            % Inputs:
            %         layer - Layer to forward propagate through 
            %         X     - Input data
            % Outputs:
            %         Z     - Output of layer forward function
            %         state - (Optional) Updated layer state

            % Define layer predict function here.
            % Implementation using fft2
            isdlarray = isa(X,'dlarray');
            if (isdlarray)
                X = extractdata(X);
            end

            Z = abs(fft2(X));

            if (isdlarray)
                Z = dlarray(Z);
            end
            
            % Implementation using fft
            % fft2 is equivalent to computing fft(fft(X).').'
            % perm = 1:numel(size(X));
            % perm([1 2]) = perm([2 1]);
            % Z = permute(fft(X),perm);
            % Z = permute(fft(Z),perm);
            % Z = abs(Z);
        end
    end
end
