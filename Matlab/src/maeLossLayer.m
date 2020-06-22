%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If you use this code, please cite the following paper:
% Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith 
% Punnappurath, and Michael S Brown. CIE XYZ Net: Unprocessing Images for 
% Low-Level Computer Vision Tasks. arXiv preprint, 2020.
%
% Author: Mahmoud Afifi | Email: mafifi@eecs.yorku.ca, m.3afifi@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef maeLossLayer < nnet.layer.RegressionLayer
    properties
        scale
    end
    methods
        function layer = maeLossLayer(name, scaleFactor)
            layer.Name = name;
            layer.Description = 'Mean absolute error';
            layer.scale = scaleFactor;
        end
        
        function loss = forwardLoss(layer, Y, T)
            % Calculate MAE.
            Y(Y<0) = 0;
            
            sz = size(Y);
            if length(sz) == 3
                R = 1;
            else
                R = sz(4);
            end
            if R == 1
                loss = abs(Y(:,:,4:6)-T(:,:,4:6)) + layer.scale * ...
                    abs(Y(:,:,1:3)-T(:,:,1:3));
            else
                loss = abs(Y(:,:,4:6,:)-T(:,:,4:6,:)) + layer.scale * ...
                    abs(Y(:,:,1:3,:)-T(:,:,1:3,:));
            end
            
            loss = sum(loss(:))/R;
        end
    end
end