
% Compile MatConvNet (only once)
run matlab/vl_compilenn
% setup MatConvNet
run matlab/vl_setupnn

% Load image data
f = filesep;
load(strcat('ImageData', f, 'myfaces.mat'));
load(strcat('ImageData', f, 'imdb.mat'));

% Add paths to execute the tests
addpath('Executions');
addpath('Pyramid');
addpath('bsif_code_and_data');
addpath(strcat('bsif_code_and_data', f, 'texturefilters'));

nfolds = 5;


%% Execution with the original images

% Create the partitions
CVO = cvpartition(images.labels, 'k', nfolds);

myMatrixHOG = HOG_General(faces, images, 6, 19, 6, 25, CVO);

% LBP
myMatrixLBP = LBP_General(faces, images, 2, 6, 2, 7, CVO);

% Create the half and quart images and concatenate them
newfaces = resize_half_and_quart(faces, images);

% LBP of concatenation of image and half image
myMatrixLBP_half = LBP_half_General(faces, newfaces, images, 2, 6, 2, 7, CVO);

% LBP of concatenation of image, half image and quarter image
myMatrixLBP_quart = LBP_quart_General(faces, newfaces, images, 2, 6, 2, 7, CVO);

% Execution of LBP Pyramid
myMatrixLBPPyramid = LBP_of_pyramid_General(5, 16, faces, images, 2, 6, 2, 7, CVO);

% BSIF
% This has to be done on 2 steps because the 3x3 filters have less bits
myMatrixBSIF = BSIF_General(faces, images, 3, 3, 5, 8, CVO);
myMatrixBSIF2 = BSIF_General(faces, images, 5, 11, 5, 12, CVO);

%% Executions with cropped and centered images
load(strcat('ImageData', f, 'centimdb.mat'));

% In the cropped images we use the same partitions (CVO)

%HOG
myMatrixHOGcent = HOG_General(faces, centimages, 6, 19, 6, 25, CVO);

%LBP
myMatrixLBPcent = LBP_General(faces, centimages, 2, 6, 2, 7, CVO);

% Create the half and quart images and concatenate them
newfacescent = resize_half_and_quart(faces, centimages);

% LBP of concatenation of image and half image
myMatrixLBPcent_half = LBP_half_General(faces, newfacescent, centimages, 2, 6, 2, 7, CVO);

% LBP of concatenation of image, half image and quarter image
myMatrixLBPcent_quart = LBP_quart_General(faces, newfacescent, centimages, 2, 6, 2, 7, CVO);

% LBP of pyramid
myMatrixLBPcentPyramid = LBP_of_pyramid_General(5, 16, faces, centimages, 2, 6, 2, 7, CVO);

% BSIF
% This has to be done on 2 steps because the 3x3 filters have less bits
myMatrixBSIFcent = BSIF_General(faces, centimages, 3, 3, 5, 8, CVO);
myMatrixBSIFcent2 = BSIF_General(faces, centimages, 5, 11, 5, 12, CVO);
