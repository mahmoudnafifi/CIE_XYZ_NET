%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If you use this code, please cite the following paper:
% Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith 
% Punnappurath, and Michael S Brown. CIE XYZ Net: Unprocessing Images for 
% Low-Level Computer Vision Tasks. arXiv preprint, 2020.
%
% Author: Mahmoud Afifi | Email: mafifi@eecs.yorku.ca, m.3afifi@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef additionLayer < nnet.layer.Layer
    
    properties 
        sign
    end
    
    methods
        function layer = additionLayer(numInputs,name, sign) 
          
            layer.NumInputs = numInputs;
            layer.Name = name;
            layer.Description = "Local image mapping";
            layer.sign = sign;
        end
        
        function Z = predict(layer, varargin)
            
            residuals = tanh(varargin{1})/4;  %limit the range to be from -0.25 to +0.25
            images = varargin{2};
            Z = images + layer.sign * residuals;
        end
    end
end