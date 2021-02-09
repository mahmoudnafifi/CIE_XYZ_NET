%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Released under the MIT License.
% If you use this code, please cite the following paper:
% Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith 
% Punnappurath, and Michael S Brown. CIE XYZ Net: Unprocessing Images for 
% Low-Level Computer Vision Tasks. arXiv preprint, 2020.
%
% Author: Mahmoud Afifi | Email: mafifi@eecs.yorku.ca, m.3afifi@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Out = applyGlobalMapping(net, I, device, pp, opt)

if nargin == 2
    device = 'gpu';
    pp = 'none';
    opt = [];
elseif nargin == 3
    pp = 'none';
    opt = [];
elseif nargin == 4
    opt = [];
end

if strcmpi(device,'gpu')
    m = gather(extractdata(predict(net, gpuArray(dlarray(I,'SSC')))));
elseif strcmpi(device,'cpu')
    m = extractdata(predict(net, dlarray(I,'SSC')));
else
    error("Wrong device value -- it should be either 'gpu' or 'cpu'");
end
m = reshape(m, [6,3]);

Out = reshape(phi(reshape(I,[],3)) * m,size(I));
if strcmp(pp,'none') ~= 1
    if strcmp(pp,'transfer') == 1
        opt.net = net;
    end
    Out = postprocessing(Out,pp, opt);
end
end
