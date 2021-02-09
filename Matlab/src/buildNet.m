%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Released under the MIT License.
% If you use this code, please cite the following paper:
% Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith 
% Punnappurath, and Michael S Brown. CIE XYZ Net: Unprocessing Images for 
% Low-Level Computer Vision Tasks. arXiv preprint, 2020.
%
% Author: Mahmoud Afifi | Email: mafifi@eecs.yorku.ca, m.3afifi@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function lgraph = buildNet(localBockDepth, localBlock_convDepth, ...
    globalBlockDepth, globalBlock_convDepth, global_trg_img_sz,...
    inputSize, loss, scaleFactor)

stgOnePrefix = 'sRGB Local Mapping';
stgTwoPrefix = 'sRGB Global Mapping';

stgThreePrefix = 'XYZ Global Mapping';
stgFourPrefix = 'XYZ Local Mapping';

inputLayer = imageInputLayer([inputSize inputSize 3],'Name',...
    'Input Layer','Normalization','none'); % no zero normalization

%% sRGB local mapping
sRGB_local_mapping_layers = [inputLayer];
sRGB_local_mapping_layers = addLocalBlock(sRGB_local_mapping_layers, ...
    localBockDepth, localBlock_convDepth, -1, stgOnePrefix);
    

%% sRGB global mapping
sRGB_global_mapping_layers = addGlobalBlock([], globalBlockDepth, ...
    globalBlock_convDepth, global_trg_img_sz, stgTwoPrefix);


%% XYZ global mapping
XYZ_global_mapping_layers = addGlobalBlock([], globalBlockDepth, ...
    globalBlock_convDepth, global_trg_img_sz, stgThreePrefix);


%% XYZ local mapping
XYZ_local_mapping_layers = addLocalBlock([], localBockDepth, ...
    localBlock_convDepth, 1, stgFourPrefix);
    

%% Loss layer
lossLayer = addLossLayer('Loss', loss, scaleFactor);

%% Concatination layers
dim = 3; %3rd dim
catLayer = concatenationLayer(dim,2,'Name','catLayer');
%srtLayer = sortLayer('sortLayer');

lgraph = layerGraph(sRGB_local_mapping_layers);

lgraph = addLayers(lgraph, sRGB_global_mapping_layers);

lgraph = addLayers(lgraph, XYZ_global_mapping_layers);

lgraph = addLayers(lgraph, XYZ_local_mapping_layers);

lgraph = connectLayers(lgraph, inputLayer.Name, ...
    [sRGB_local_mapping_layers(end).Name '/in2']);

lgraph = connectLayers(lgraph, sRGB_local_mapping_layers(end).Name, ...
    sRGB_global_mapping_layers(1).Name);

lgraph = connectLayers(lgraph, sRGB_global_mapping_layers(1).Name, ...
    [sRGB_global_mapping_layers(end).Name '/in2']);

lgraph = connectLayers(lgraph, sRGB_global_mapping_layers(end).Name, ...
    XYZ_global_mapping_layers(1).Name);


lgraph = connectLayers(lgraph, sRGB_global_mapping_layers(end).Name, ...
    [XYZ_global_mapping_layers(end).Name '/in2']);

lgraph = connectLayers(lgraph, XYZ_global_mapping_layers(end).Name, ...
    XYZ_local_mapping_layers(1).Name);

lgraph =  addLayers(lgraph, [catLayer, lossLayer]);

lgraph = connectLayers(lgraph, XYZ_global_mapping_layers(end).Name, ...
    [XYZ_local_mapping_layers(end).Name '/in2']);

lgraph = connectLayers(lgraph, XYZ_local_mapping_layers(end).Name, ...
    [catLayer.Name '/in2']);

lgraph = connectLayers(lgraph, sRGB_global_mapping_layers(end).Name, ...
    [catLayer.Name '/in1']);