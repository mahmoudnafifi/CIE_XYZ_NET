%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Released under the MIT License.
% If you use this code, please cite the following paper:
% Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith 
% Punnappurath, and Michael S Brown. CIE XYZ Net: Unprocessing Images for 
% Low-Level Computer Vision Tasks. arXiv preprint, 2020.
%
% Author: Mahmoud Afifi | Email: mafifi@eecs.yorku.ca, m.3afifi@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function nets = getSubNetworks(net, inputSize)

net = layerGraph(net);

if inputSize(3)~=3
    error('Cannot process grayscale images');
end

inputLayer = imageInputLayer([inputSize(1) inputSize(2) inputSize(3)], ...
    'Name', 'Input Layer','Normalization','none'); % no zero normalization

local_mp_sRGB = 'sRGB Local Mapping';
local_mp_XYZ = 'XYZ Local Mapping';
global_mp_sRGB = 'sRGB Global Mapping';
global_mp_XYZ = 'XYZ Global Mapping';

layerNames = {net.Layers.Name}';

%% get local sRGB subnet
nets.local_sRGB = net.Layers(contains(layerNames,local_mp_sRGB));

nets.local_sRGB = [inputLayer; nets.local_sRGB(1:end-1)]; ...


%% get local XYZ subnet
nets.local_XYZ = net.Layers(contains(layerNames,local_mp_XYZ));

nets.local_XYZ = [inputLayer; nets.local_XYZ(1:end-1)]; ...


%% get global sRGB subnet
nets.global_sRGB = net.Layers(contains(layerNames,global_mp_sRGB));

nets.global_sRGB = [inputLayer; nets.global_sRGB(2:end-1)]; ...

%% get global XYZ subnet
nets.global_XYZ = net.Layers(contains(layerNames,global_mp_XYZ));

nets.global_XYZ = [inputLayer; nets.global_XYZ(2:end-1)]; ...
    
%% convert sub-networks into dlnetworks
nets.local_sRGB = dlnetwork(layerGraph(nets.local_sRGB));
nets.local_XYZ = dlnetwork(layerGraph(nets.local_XYZ));
nets.global_sRGB = dlnetwork(layerGraph(nets.global_sRGB));
nets.global_XYZ = dlnetwork(layerGraph(nets.global_XYZ));

end

