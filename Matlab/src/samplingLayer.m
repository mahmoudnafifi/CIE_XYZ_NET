%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Released under the MIT License.
% If you use this code, please cite the following paper:
% Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith 
% Punnappurath, and Michael S Brown. CIE XYZ Net: Unprocessing Images for 
% Low-Level Computer Vision Tasks. arXiv preprint, 2020.
%
% Author: Mahmoud Afifi | Email: mafifi@eecs.yorku.ca, m.3afifi@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef samplingLayer < nnet.layer.Layer
    
    properties
        targetSize
    end
    
    methods
        function layer = samplingLayer(name, targetSize)
            
            layer.Name = name;
            layer.Description = "sampling layer";
            layer.targetSize = targetSize;
            
        end
        
        function Z = predict(layer, X)
          inds_1 = round(linspace(1,size(X,1),layer.targetSize));
          inds_2 = round(linspace(1,size(X,2),layer.targetSize));
          if length(size(X)) == 4
              Z = X(inds_1,inds_2,:,:);
          else
              Z = X(inds_1,inds_2,:);
          end
        end
    end
end

