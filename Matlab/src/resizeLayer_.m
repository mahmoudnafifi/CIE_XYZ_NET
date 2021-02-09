%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Released under the MIT License.
% If you use this code, please cite the following paper:
% Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith
% Punnappurath, and Michael S Brown. CIE XYZ Net: Unprocessing Images for
% Low-Level Computer Vision Tasks. arXiv preprint, 2020.
%
% Author: Mahmoud Afifi | Email: mafifi@eecs.yorku.ca, m.3afifi@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef resizeLayer_ < nnet.layer.Layer
    
    properties
        target_size
    end
    
    methods
        function layer = resizeLayer_(name, target_size)
            layer.Name = name;
            layer.Description = "Image resizing";
            layer.target_size = target_size;
        end
        
        function Z = predict(layer, X)
            sz = size(X);
            if length(sz) < 3
                sz = [sz ones(1, 3 - length(sz))];
                
            end
            poolsize_d1 = floor(sz(1)/layer.target_size);
            poolsize_d2 = floor(sz(2)/layer.target_size);
            
            if isempty(X) || sz(1) == 0
                Z = X;
            else
                L = length(X(:))/(sz(1) * sz(2) * sz(3));
                if L == 1
                    Z = zeros(layer.target_size, layer.target_size, ...
                        sz(3), 'like',X);
                    temp = avgpool(X,[poolsize_d1, poolsize_d2], ...
                        'Stride', [poolsize_d1, poolsize_d2],...
                        'DataFormat','SSCB');
                    Z = temp(1:layer.target_size,1:layer.target_size,:);
                    size(Z)
                else
                    Z = zeros(layer.target_size, layer.target_size, ...
                        sz(3), sz(4), 'like', X);
                    for i = 1: L
                        temp = avgpool(X(:,:,:,i),...
                            [poolsize_d1, poolsize_d2],...
                            'Stride', [poolsize_d1, poolsize_d2],...
                            'DataFormat','SSCB');
                        Z(:,:,:,i) = ...
                            temp(1:layer.target_size, ...
                            1:layer.target_size,:,i);
                    end
                end
            end
        end
    end
end

