classdef windowingLayer < nnet.layer.Layer & nnet.layer.Acceleratable
        % & nnet.layer.Formattable ... % (Optional) 

    properties (Learnable)
        W
    end

    properties (State)
        % (Optional) Layer state parameters.

        % Declare state parameters here.
    end

    methods
        function layer = windowingLayer(args)
            arguments
                args.Name = "";
            end

            layer.Name = args.Name;
        end
        
        function layer = initialize(layer,layout)
            layer.W = ones(layout.Size([1 2 3]));
        end

        function Z = predict(layer,X)
            Z = X .* layer.W;
        end
    end
end