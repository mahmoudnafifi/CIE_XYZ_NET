%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Please adjust this function with your custom functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function O = postprocessing(I, PP, opt)

if nargin == 2
    opt = [];
end

if contains(PP,'+')
    PP = strsplit(PP,'+');
else
    PP = {PP};
end

O = I;

for i = 1 : length(PP)
    
    pp = PP{i};
    switch pp
        case 'denoise'
            disp('denoising...');
            % denoising code goes here
        case 'chrom-adapt'
            disp('chrom adapting...');
            % chromatic adaptation code goes here
        case 'deblur'
            disp('deblurring...');
            % deblurring code goes here
        case 'dehaze'
            disp('dehazing...');
            % dehazing code goes here
            
        case 'editdetails' 
            disp('editing local details...');
            % local detail enhancement code goes here
            
        case 'exposure-fusion'
            disp('exposure fusion...');
            % exposure fusion code goes here
            
        case 'transfer-colors' 
            disp('color transfering...');
            % color transfer code goes here
            % you may need to use the 'opt' variable here
            
        case 'super-res' 
            disp('super-resolution processing...');
            % super resolution code goes here
            
        otherwise
            disp('wrong post-processing task!');
    end
end
end

