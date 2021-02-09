%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Released under the MIT License.
% If you use this code, please cite the following paper:
% Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith 
% Punnappurath, and Michael S Brown. CIE XYZ Net: Unprocessing Images for 
% Low-Level Computer Vision Tasks. arXiv preprint, 2020.
%
% Author: Mahmoud Afifi | Email: mafifi@eecs.yorku.ca, m.3afifi@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% trainig code

clc
clear;
close all;

fprintf('Training code\n');

localBockDepth = 16; % # of [conv/relu] blocks for the local processing branch 

localBlock_convDepth = 32; % depth of each conv layer in local branch

globalBlockDepth = 5; % # of [conv/relu/2x2 avgpool] blocks in global processisng

globalBlock_convDepth = 64; % depth of each conv layer in global branch

global_trg_img_sz = 128; % resizing size in the global branch

inputSize = 256; % original traning patch size

loss = 'mae'; 

load_model = 0; %resume training? then provide the full path in model_path

model_path = '';

checkpoint_dir = ['reports_and_backup_' ...
    strrep(strrep(char(datetime),':','-'),' ','_')]; % to save reports and bkups 

GPUDevice = 1; % which gpu device?

L2Reg = 10^-3; % L2 regularization factor

epochs = 300;

miniBatch = 4;

lR = 10^-4; %learning rate

checkpoint_period = 20; % period to take a backup model

validation_frequency = 1000; %every 1000 iterations, validate

modelName = 'model_sRGB-XYZ-sRGB.mat'; % the trained model filename

fprintf('Prepare training data ...\n');

TrsRGB_dir = fullfile('..','sRGB_training');

TrXYZ_dir = fullfile('..','XYZ_training');

VlsRGB_dir = fullfile('..','sRGB_validation');

VlXYZ_dir = fullfile('..','XYZ_validation');

aug = 1; %apply geometric augmentation?

Tr_ImageNum = 0; %number of training patches, use 0 to load all

Vl_ImageNum = 0; %similarly here

scaleFactor = 1.5; %scale XYZ by this factor

[Trdata,Vldata] = getTrVlData(TrsRGB_dir, TrXYZ_dir, VlsRGB_dir, ...
    VlXYZ_dir, aug, Tr_ImageNum, Vl_ImageNum, [inputSize inputSize]);


options = get_training_options(epochs,miniBatch,lR,...
    checkpoint_dir,Vldata, validation_frequency, GPUDevice, L2Reg, ...
    checkpoint_period); % training options


if load_model == 1
    fprintf('Uploading the model ...\n');
    load(model_path)
    net = layerGraph(net);
else
    fprintf('Creating the model ...\n');
    net = buildNet(localBockDepth, localBlock_convDepth, ...
    globalBlockDepth, globalBlock_convDepth, global_trg_img_sz,...
    inputSize, loss, scaleFactor); %build the model
end

fprintf('Start training ...\n');

net = trainNetwork(Trdata,net,options); % start training ...

fprintf('Saving the trained model ...\n');

if exist('models','dir') == 0
    mkdir('models');
end

nets = getSubNetworks(net, [inputSize, inputSize, 3]);

save(fullfile('models',modelName),'nets','-v7.3'); % save the trained model
