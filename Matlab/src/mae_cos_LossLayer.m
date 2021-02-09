%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Released under the MIT License.
% If you use this code, please cite the following paper:
% Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith 
% Punnappurath, and Michael S Brown. CIE XYZ Net: Unprocessing Images for 
% Low-Level Computer Vision Tasks. arXiv preprint, 2020.
%
% Author: Mahmoud Afifi | Email: mafifi@eecs.yorku.ca, m.3afifi@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef mae_cos_LossLayer < nnet.layer.RegressionLayer
    properties
        scale
    end
    methods
        function layer = mae_cos_LossLayer(name, scaleFactor)
            layer.Name = name;
            layer.Description = ...
                'Mean abs error + (negative) cos similarity';
            layer.scale = scaleFactor;
        end
        
        function loss = forwardLoss(layer, Y, T)
            % Calculate MAE + (negative) cos similarity.
            Y(Y<0) = 0;
            
            sz = size(Y);
            if length(sz) == 3
                R = 1;
                mae_srgb = sum(reshape(abs(Y(:,:,4:6) - T(:,:,4:6)),...
                    [],3),2);
                mae_xyz = sum(reshape(abs(Y(:,:,1:3) - T(:,:,1:3)),...
                    [],3),2);
                
                % compute cosine simialrity
                cosSim_srgb = zeros(size(mae_srgb), 'like', Y);
                cosSim_xyz = zeros(size(mae_xyz), 'like', Y);
                
                y_srgb = reshape(Y(:,:,3:6),[],3);
                t_srgb = reshape(T(:,:,4:6),[],3);
                y_xyz = reshape(Y(:,:,1:3),[],3);
                t_xyz = reshape(T(:,:,1:3),[],3);
                
                cosSim_srgb = cosSim_srgb + sum(y_srgb.*t_srgb,2)./...
                    (sqrt(sum(y_srgb.^2,2)) .* ...
                    sqrt(sum(t_srgb.^2,2)) + eps);
                
                cosSim_xyz = cosSim_xyz + sum(y_xyz.*t_xyz,2)./...
                    (sqrt(sum(y_xyz.^2,2)) .* ...
                    sqrt(sum(t_xyz.^2,2)) + eps);
                
            else
                R = sz(4);
                mae_srgb = zeros(sz(1)*sz(2),1,'like',Y);
                cosSim_srgb = zeros(size(mae_srgb),'like',mae_srgb);
                
                mae_xyz = zeros(sz(1)*sz(2),1,'like',Y);
                cosSim_xyz = zeros(size(mae_xyz),'like',mae_xyz);
                
                for j = 1 : R
                    
                    mae_srgb = mae_srgb + sum(reshape( ...
                        abs(Y(:,:,4:6,j) - T(:,:,4:6,j)) ,...
                        [],3),2);
                    mae_xyz = mae_xyz + sum(reshape( ...
                        abs(Y(:,:,1:3,j) - T(:,:,1:3,j)) ,...
                        [],3),2);
                    
                    % compute cosine simialrity
                    y_srgb = reshape(Y(:,:,4:6),[],3);
                    t_srgb = reshape(T(:,:,4:6),[],3);
                    y_xyz = reshape(Y(:,:,1:3),[],3);
                    t_xyz = reshape(T(:,:,1:3),[],3);
                    
                    cosSim_srgb = cosSim_srgb + sum(y_srgb.*t_srgb,2)./...
                        (sqrt(sum(y_srgb.^2,2)) .* ...
                        sqrt(sum(t_srgb.^2,2)) + eps);
                    
                    cosSim_xyz = cosSim_xyz + sum(y_xyz.*t_xyz,2)./...
                        (sqrt(sum(y_xyz.^2,2)) .* ...
                        sqrt(sum(t_xyz.^2,2)) + eps);
                end
            end
            
            loss = (sum(mae_srgb - cosSim_srgb) + ...
                layer.scale * sum(mae_xyz - cosSim_xyz))/R;
            
        end
    end
end