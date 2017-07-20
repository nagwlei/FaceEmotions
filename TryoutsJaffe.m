
% Compile MatConvNet (only once)
run matlab/vl_compilenn
% setup MatConvNet
run matlab/vl_setupnn

% Load image data
f = filesep;
load(strcat('ImageData', f, 'jaffeimdb.mat'));

addpath('Executions');
addpath('Pyramid');
addpath('bsif_code_and_data');
addpath(strcat('bsif_code_and_data', f, 'texturefilters'));

% N folds = Number of different people in the db
nfolds = 10;

%% Execution with original Jaffe db images
if (~exist('CVOjaffe'))
    mapObj = containers.Map({'KA', 'KL', 'KM', 'KR', 'MK', 'NA','NM', 'TM', 'UY', 'YM'}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    names = zeros(1, length( facesjaffe ) );
    for kk=1:length(facesjaffe)
        names(kk) = mapObj(facesjaffe{kk}.id);
    end
    
    CVOjaffe = cvpartition(names, 'k', nfolds);
end

% HOG
myMatrixHOGjaffe = HOG_General(facesjaffe, jaffeimgs, 6, 19, 6, 25, CVOjaffe);

% LBP
myMatrixLBPjaffe = LBP_General(facesjaffe, jaffeimgs, 2, 6, 2, 7, CVOjaffe);

% Create the half and quart images and concatenate them
newfacesjaffe = resize_half_and_quart(facesjaffe, jaffeimgs);

% LBP of concatenation of image and half image
myMatrixLBP_halfjaffe = LBP_half_General(facesjaffe, newfacesjaffe, jaffeimgs, 2, 6, 2, 7, CVOjaffe);

% LBP of concatenation of image, half image and quarter image
myMatrixLBP_quartjaffe = LBP_quart_General(facesjaffe, newfacesjaffe, jaffeimgs, 2, 6, 2, 7, CVOjaffe);

% Execution of LBP Pyramid
myMatrixLBPPyramidjaffe = LBP_of_pyramid_General(5, 16, facesjaffe, jaffeimgs, 2, 6, 2, 7, CVOjaffe);

% BSIF
% This has to be done on 2 steps because the 3x3 filters have less bits
myMatrixBSIFjaffe = BSIF_General(facesjaffe, jaffeimgs, 3, 3, 5, 8, CVOjaffe);
myMatrixBSIF2jaffe = BSIF_General(facesjaffe, jaffeimgs, 5, 11, 5, 12, CVOjaffe);

%BSIF of concatenation of image and half image
myMatrixBSIF_halfjaffe = BSIF_half_General(facesjaffe, newfacesjaffe, jaffeimgs, 3, 3, 5, 8, CVOjaffe);
myMatrixBSIF_half2jaffe = BSIF_half_General(facesjaffe, newfacesjaffe, jaffeimgs, 5, 11, 5, 12, CVOjaffe);

% Save results in a 'oridbresults.mat'
save('orijafferesults.mat', 'CVOjaffe', 'myMatrixHOGjaffe', 'myMatrixLBPjaffe', ...
    'myMatrixLBP_halfjaffe', 'myMatrixLBP_quartjaffe', 'myMatrixLBPPyramidjaffe', ...
    'myMatrixBSIFjaffe', 'myMatrixBSIF2jaffe', 'myMatrixBSIF_halfjaffe', ...
    'myMatrixBSIF_half2jaffe')
