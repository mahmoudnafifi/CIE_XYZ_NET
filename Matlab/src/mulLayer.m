%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If you use this code, please cite the following paper:
% Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith 
% Punnappurath, and Michael S Brown. CIE XYZ Net: Unprocessing Images for 
% Low-Level Computer Vision Tasks. arXiv preprint, 2020.
%
% Author: Mahmoud Afifi | Email: mafifi@eecs.yorku.ca, m.3afifi@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef mulLayer < nnet.layer.Layer
    
    properties
        
    end
    
    methods
        function layer = mulLayer(numInputs,name)
            
            layer.NumInputs = numInputs;
            layer.Name = name;
            layer.Description = "Global image mapping";
        end
        
        function Z = predict(layer, varargin)
            mappingFuncs = varargin{1};
            images = varargin{2};
            sz = size(images);
            if length(sz) < 3
                Z = images;
            else
                Z = zeros(sz,'like',images);
                if length(sz) == 3
                    Z = reshape(...
                        phi(reshape(images,[sz(1)*sz(2),3])) * ...
                        reshape(mappingFuncs,[length(mappingFuncs(:))/3,...
                        3]), [sz(1), sz(2), sz(3)]);
                else
                    L = sz(4);
                    for i = 1:L
                        m = mappingFuncs(:,:,:,i);
                        Z(:,:,:,i) = reshape(...
                            phi(reshape(images(:,:,:,i),[sz(1)*sz(2),3])) * ...
                            reshape(m,[length(m(:))/3,3]), ...
                            [sz(1), sz(2), sz(3)]);
                    end
                end
            % sprintf('Mul-redisuals:')
            %size(Z)
            
            end
        end
    end
end