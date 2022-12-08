classdef fft2Layer < nnet.layer.Layer

    methods
        function layer = fft2Layer(NameValueArgs) 
            % layer = fft2Layer(numInputs,name) creates a
            % fft2 layer and specifies the layer name.

            arguments
                NameValueArgs.Name
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
            %
            %  - For layers with multiple inputs, replace X with X1,...,XN, 
            %    where N is the number of inputs.
            %  - For layers with multiple outputs, replace Z with 
            %    Z1,...,ZM, where M is the number of outputs.
            %  - For layers with multiple state parameters, replace state 
            %    with state1,...,stateK, where K is the number of state 
            %    parameters.

            % Define layer predict function here.
            Z = fft2(X);
        end
    end
end
