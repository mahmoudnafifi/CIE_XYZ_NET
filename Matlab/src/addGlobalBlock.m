%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Released under the MIT License.
% If you use this code, please cite the following paper:
% Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith 
% Punnappurath, and Michael S Brown. CIE XYZ Net: Unprocessing Images for 
% Low-Level Computer Vision Tasks. arXiv preprint, 2020.
%
% Author: Mahmoud Afifi | Email: mafifi@eecs.yorku.ca, m.3afifi@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function layers = addGlobalBlock(layers, blockDepth, convDepth, ...
    trgt_img_sz, prefix)

layers = [debugLayer([prefix 'debug']), layers, ...
    resizeLayer([prefix ' Resizing'],trgt_img_sz)];

for layerNumber = 1 : blockDepth
    
    convolutionLayer = convolution2dLayer(3,convDepth, ...
        'Padding',[1 1], 'WeightsInitializer','he', ...
        'BiasInitializer', 'zeros', 'Name',[prefix ' Conv' ...
        num2str(layerNumber)]);
    relLayer = leakyReluLayer('Name',[prefix ' L-ReLU' ...
        num2str(layerNumber)]);
    poolLayer = maxPooling2dLayer(2, 'Stride', 2, 'Padding', 0, ...
        'Name', [prefix ' maxPooling' num2str(layerNumber)]);
    layers = [layers convolutionLayer relLayer poolLayer];
end
fc1 = fullyConnectedLayer(1024,'Name',[prefix ' fc1']);
dout = dropoutLayer(0.5, 'Name',[prefix ' dropout']);
fcout = fullyConnectedLayer(3 * 6, 'Name', [prefix 'fc-out']);
mul_layer = mulLayer (2, [prefix ' Global-processing']);
layers = [layers fc1 dout fcout mul_layer];
end

