
% Compile MatConvNet (only once)
%run matlab/vl_compilenn
% setup MatConvNet
%run matlab/vl_setupnn

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
    mapObj = containers.Map({'KA', 'KL', 'KM', 'KR', 'MK', 'NA','NM', ...
        'TM', 'UY', 'YM'}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    CVOjaffe.NumTestSets = nfolds;
    CVOjaffe.training = cell(1, nfolds);
    CVOjaffe.test = cell(1, nfolds);
    CVOjaffe.TrainSize = zeros(1, nfolds);
    CVOjaffe.TestSize = zeros(1, nfolds);
    CVOjaffe.NumObservations = length(facesjaffe);
    
    theIds = keys(mapObj)

    classesIndexes = zeros(1, length(facesjaffe));

    for kk=1:nfolds
        kk
        aux = theIds(kk);
        ispresent = cellfun(@(s) ~isempty(strfind(aux{1}, s.id)), facesjaffe);

        %foldIndexes{1,kk} = ispresent;
        CVOjaffe.TestSize(kk) = sum(ispresent);
        CVOjaffe.test{kk} = ispresent;
        CVOjaffe.TrainSize(kk) = sum(~CVOjaffe.test{kk});
        CVOjaffe.training{kk} = ~CVOjaffe.test{kk};
    end;
end

% HOG
myMatrixHOGjaffe = HOG_General(facesjaffe, jaffeimgs, 6, 19, 6, 25, ...
    CVOjaffe);

% LBP
myMatrixLBPjaffe = LBP_General(facesjaffe, jaffeimgs, 2, 6, 2, 7, ...
    CVOjaffe);

% Create the half and quart images and concatenate them
newfacesjaffe = resize_half_and_quart(facesjaffe, jaffeimgs);

% LBP of concatenation of image and half image
myMatrixLBP_halfjaffe = LBP_half_General(facesjaffe, newfacesjaffe, ...
    jaffeimgs, 2, 6, 2, 7, CVOjaffe);

% LBP of concatenation of image, half image and quarter image
myMatrixLBP_quartjaffe = LBP_quart_General(facesjaffe, newfacesjaffe, ...
    jaffeimgs, 2, 6, 2, 7, CVOjaffe);

% Execution of LBP Pyramid
myMatrixLBPPyramidjaffe = LBP_of_pyramid_General(5, 16, facesjaffe, ...
    jaffeimgs, 2, 6, 2, 7, CVOjaffe);

% BSIF
% This has to be done on 2 steps because the 3x3 filters have less bits
myMatrixBSIFjaffe = BSIF_General(facesjaffe, jaffeimgs, 3, 3, 5, 8, ...
    CVOjaffe);
myMatrixBSIF2jaffe = BSIF_General(facesjaffe, jaffeimgs, 5, 11, 5, 12, ...
    CVOjaffe);

%BSIF of concatenation of image and half image
myMatrixBSIF_halfjaffe = BSIF_half_General(facesjaffe, newfacesjaffe, ...
    jaffeimgs, 3, 3, 5, 8, CVOjaffe);
myMatrixBSIF_half2jaffe = BSIF_half_General(facesjaffe, newfacesjaffe, ...
    jaffeimgs, 5, 11, 5, 12, CVOjaffe);

%BSIF of concatenation of image, half and quart image
myMatrixBSIF_quartjaffe = BSIF_quart_General(facesjaffe, newfacesjaffe, ...
    jaffeimgs, 3, 3, 5, 8, CVOjaffe);
myMatrixBSIF_quart2jaffe = BSIF_quart_General(facesjaffe, newfacesjaffe, ...
    jaffeimgs, 5, 11, 5, 12, CVOjaffe);

HybridMAE = Hybrid_LBP_HOG_BSIFT(facesjaffe, jaffeimgs, myMatrixLBPjaffe, ...
    2, 2, myMatrixHOGjaffe, 6, 6, 5, 5, myMatrixBSIFjaffe, ...
    myMatrixBSIF2jaffe, CVOjaffe);


HybridMAE_concat = Hybrid_LBP_HOG_BSIFT_Pyramid(facesjaffe, newfacesjaffe, ...
    jaffeimgs, myMatrixLBPjaffe, myMatrixLBP_halfjaffe, myMatrixLBP_quartjaffe,... 
    myMatrixLBPPyramidjaffe, 2, 2, myMatrixHOGjaffe, ...
    6, 6, 5, ...
    myMatrixBSIFjaffe, myMatrixBSIF2jaffe, myMatrixBSIF_halfjaffe, myMatrixBSIF_half2jaffe, ...
    myMatrixBSIF_quartjaffe, myMatrixBSIF_quart2jaffe, CVOjaffe);

% Save results in a 'oridbresults.mat'
save('oriJAFFEresults.mat', 'CVOjaffe', 'myMatrixHOGjaffe', ...
    'myMatrixLBPjaffe', 'myMatrixLBP_halfjaffe', ...
    'myMatrixLBP_quartjaffe', 'myMatrixLBPPyramidjaffe', ...
    'myMatrixBSIFjaffe', 'myMatrixBSIF2jaffe', ...
    'myMatrixBSIF_halfjaffe', 'myMatrixBSIF_half2jaffe', ...
    'HybridMAE', 'HybridMAE_concat')
