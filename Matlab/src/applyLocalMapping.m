%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If you use this code, please cite the following paper:
% Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith 
% Punnappurath, and Michael S Brown. CIE XYZ Net: Unprocessing Images for 
% Low-Level Computer Vision Tasks. arXiv preprint, 2020.
%
% Author: Mahmoud Afifi | Email: mafifi@eecs.yorku.ca, m.3afifi@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Out] = applyLocalMapping(net, I, target, device, pp, opt)

if nargin == 3
    device = 'gpu';
    pp = 'none';
    opt = [];
elseif nargin == 4
    pp = 'none';
    opt = [];
elseif nargin == 5
    opt = [];
end

switch target
    case 'to-xyz'        
        if strcmpi(device,'gpu')
            res = tanh(gather(extractdata(predict(net, ...
                gpuArray(dlarray(I,'SSC'))))))/4;
        elseif strcmpi(device,'cpu')
            res = tanh(extractdata(predict(net, dlarray(I,'SSC'))))/4;
        else
            error("Wrong device value -- it should be either 'gpu' or 'cpu'");
        end
        
        
        if strcmp(pp,'none') ~= 1
            res = postprocessing(res,pp, opt);
        end
        Out = I - res;
    case 'to-srgb'
        if strcmpi(device,'gpu')
            res = tanh(gather(extractdata(predict(net, ...
                gpuArray(dlarray(I,'SSC'))))))/4;
        elseif strcmpi(device,'cpu')
            res = tanh(extractdata(predict(net, dlarray(I,'SSC'))))/4;
        else
            error("Wrong device value -- it should be either 'gpu' or 'cpu'");
        end
        if strcmp(pp,'none') ~= 1
            res = postprocessing(res,pp, opt);
        end
        Out = I + res;
    otherwise
        error('unknown task');
end
