%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If you use this code, please cite the following paper:
% Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith 
% Punnappurath, and Michael S Brown. CIE XYZ Net: Unprocessing Images for 
% Low-Level Computer Vision Tasks. arXiv preprint, 2020.
%
% Author: Mahmoud Afifi | Email: mafifi@eecs.yorku.ca, m.3afifi@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function layers = addLocalBlock(layers, blockDepth, convDepth, sign, prefix)
for layerNumber = 1 : blockDepth
    if  layerNumber ~= blockDepth
        convolutionLayer = convolution2dLayer(3,convDepth, ...
            'Padding',[1 1], 'WeightsInitializer','he', ...
        'BiasInitializer', 'zeros', 'Name',[prefix ' Conv' ...
        num2str(layerNumber)]);
        
    else
        convolutionLayer = convolution2dLayer(3, 3, 'Padding',[1 1], ...
            'WeightsInitializer','he','BiasInitializer', 'zeros', ...
            'Name',[prefix ' Conv' num2str(layerNumber)]);
    end
    if  layerNumber ~= blockDepth
        relLayer = leakyReluLayer('Name',[prefix ' L-ReLU'...
            num2str(layerNumber)]);
        layers = [layers convolutionLayer relLayer];
    else
        localProcessingLayer = additionLayer(2, ...
            [prefix ' Local-processing'], sign);
        layers = [layers convolutionLayer localProcessingLayer];
    end
end
end

