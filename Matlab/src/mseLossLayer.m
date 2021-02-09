%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Released under the MIT License.
% If you use this code, please cite the following paper:
% Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith 
% Punnappurath, and Michael S Brown. CIE XYZ Net: Unprocessing Images for 
% Low-Level Computer Vision Tasks. arXiv preprint, 2020.
%
% Author: Mahmoud Afifi | Email: mafifi@eecs.yorku.ca, m.3afifi@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef mseLossLayer < nnet.layer.RegressionLayer
    properties
        scale
    end
    methods
        function layer = mseLossLayer(name, scaleFactor)
            layer.Name = name;
            layer.Description = 'Mean square error';
            layer.scale = scaleFactor;
        end
        
        function loss = forwardLoss(layer, Y, T)
            % Calculate MSE.
            Y(Y<0) = 0;
            
            sz = size(Y);
            if length(sz) == 3
                R = 1;
            else
                R = sz(4);
            end
            if R == 1
                loss = (Y(:,:,4:6)-T(:,:,4:6)).^2 + layer.scale * ...
                    (Y(:,:,1:3)-T(:,:,1:3)).^2;
            else
                loss = (Y(:,:,4:6,:)-T(:,:,4:6,:)).^2 + layer.scale * ...
                    (Y(:,:,1:3,:)-T(:,:,1:3,:)).^2;
            end
            
            loss = sum(loss(:))/R;
        end
    end
end