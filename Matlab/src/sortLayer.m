%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Released under the MIT License.
% If you use this code, please cite the following paper:
% Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith 
% Punnappurath, and Michael S Brown. CIE XYZ Net: Unprocessing Images for 
% Low-Level Computer Vision Tasks. arXiv preprint, 2020.
%
% Author: Mahmoud Afifi | Email: mafifi@eecs.yorku.ca, m.3afifi@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef sortLayer < nnet.layer.Layer
    
    properties
        
    end
    
    methods
        function layer = sortLayer(name)
            
            layer.Name = name;
            layer.Description = "Sorting layer (sRGB, XYZ, ... etc)";
        end
        
        function Z = predict(layer, X)
            sz = size(X);
            length(sz)
            if length(sz) == 4
                Z = zeros(sz,'like',X);
                sRGB = X(:,:, 1: 1:3,:);
                XYZ = X(:,:, 4:6, :);
                count = 1;
                for i = 1 : sz(4)
                    Z(:,:,1:3,i) = sRGB(:,:,:,count);
                    Z(:,:,4:6,i) = XYZ(:,:,:,count);
                end
            else
                Z = zeros(sz,'like',X);
                sRGB = X(:,:, 1: 1:3);
                XYZ = X(:,:, 4:6);
                Z(:,:,1:3) = sRGB;
                Z(:,:,4:6) = XYZ;
            end
               
        end
    end
end