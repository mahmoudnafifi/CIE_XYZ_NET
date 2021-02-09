%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Released under the MIT License.
% If you use this code, please cite the following paper:
% Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith 
% Punnappurath, and Michael S Brown. CIE XYZ Net: Unprocessing Images for 
% Low-Level Computer Vision Tasks. arXiv preprint, 2020.
%
% Author: Mahmoud Afifi | Email: mafifi@eecs.yorku.ca, m.3afifi@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function lossLayer = addLossLayer(name, loss, scaleFactor)

switch lower(loss)
    case 'mse'
        lossLayer = mseLossLayer(name, scaleFactor);
    case 'mae'
        lossLayer = maeLossLayer(name, scaleFactor);
    case 'mse + cosine'
        lossLayer = mse_cos_LossLayer(name, scaleFactor);
    case 'mae + cosine'
        lossLayer = mae_cos_LossLayer(name, scaleFactor);
    otherwise
        error('Wrong value in inSpace argument');
end


