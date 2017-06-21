
% Compile MatConvNet (only once)
run matlab/vl_compilenn
% setup MatConvNet
run matlab/vl_setupnn

% Load image data
load('ImageData\myfaces.mat');
load('ImageData\imdb.mat');



addpath('Executions');
%%%%%%%%%%%%%%%MISSING OTHER FUNCTIONS %%%%%%%%%%%%%%%%%%%%%
% LBP
myMatrixLBP = LBP_General(faces, images, 2, 6, 2, 7, 5);


% Execution of LBP Pyramid
addpath('Pyramid');
%myMatrixLBP = LBP_of_pyramid_General(5, 16, faces, images, 2, 6, 2, 4, 5);
myMatrixLBPPyramid = LBP_of_pyramid_General(5, 16, faces, images, 2, 7, 2, 5, 5);
myMatrixLBPPyramid = LBP_of_pyramid_General(5, 16, faces, images, 2, 6, 2, 7, 5);


%% Executions with cropped and centered images
load('ImageData\centimdb.mat');


%myMatrixHOGcent = HOG_General(faces, centimages, 6, 19, 6, 25, 5);
myMatrixHOGcent = HOG_General(faces, centimages, 6, 19, 6, 15, 5);
myMatrixHOGcent = HOG_General(faces, centimages, 6, 19, 16, 25, 5);


%myMatrixLBPcent = LBP_General(faces, centimages, 2, 6, 2, 8, 5);
%myMatrixLBPcent = LBP_General(faces, centimages, 2, 6, 2, 6, 5);
%myMatrixLBPcent2 = LBP_General(faces, centimages, 2, 5, 7, 8, 5);
%myMatrixLBPcent3 = LBP_General(faces, centimages, 5, 6, 7, 8, 5);

myMatrixLBPcent = LBP_General(faces, centimages, 2, 6, 2, 7, 5);


newfaces = resize_half_and_quart(faces, centimages);
%myMatrixLBPcent_half = LBP_half_General(faces, newfaces, centimages, 2, 6, 2, 8, 5);
myMatrixLBPcent_half = LBP_half_General(faces, newfaces, centimages, 2, 6, 2, 7, 5);


myMatrixLBPcent_quart = LBP_quart_General(faces, newfaces, centimages, 2, 6, 2, 7, 5);

myMatrixLBPcentPyramid = LBP_of_pyramid_General(5, 16, faces, centimages, 2, 6, 2, 7, 5);

addpath('bsif_code_and_data');
addpath('bsif_code_and_data\texturefilters');

myMatrixBSIF = BSIF_General(faces, centimages, 3, 3, 5, 8, 5);
myMatrixBSIF = BSIF_General(faces, images, 5, 12, 5, 12, 5);
