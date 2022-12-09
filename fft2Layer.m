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
            isdlarray = isa(X,'dlarray');
            if (isdlarray)
                X = extractdata(X);
            end

            Z = fftshift(abs(fft2(X)));

            if (isdlarray)
                Z = dlarray(Z);
            end
        end
    end
end
