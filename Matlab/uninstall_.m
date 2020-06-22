%uninstall
disp('Uninstalling...')
current = pwd;
rmpath(fullfile(current,'src'));
rmpath(fullfile(current,'pp_code'));
savepath
disp('Done!');
