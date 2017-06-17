
% Compile MatConvNet (only once)
run matlab/vl_compilenn
% setup MatConvNet
run matlab/vl_setupnn

% Load image data
load('ImageData\myfaces.mat');
load('ImageData\imdb.mat');



%%%%%%%%%%%%%%%MISSING OTHER FUNCTIONS %%%%%%%%%%%%%%%%%%%%%



% Execution of LBP Pyramid
addpath('Pyramid');
%myMatrixLBP = LBP_of_pyramid_General(5, 16, faces, images, 2, 6, 2, 4, 5);
myMatrixLBP = LBP_of_pyramid_General(5, 16, faces, images, 2, 7, 2, 5, 5);

