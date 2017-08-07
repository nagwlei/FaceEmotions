% Compile MatConvNet (only once)
%run matlab/vl_compilenn
% setup MatConvNet
%run matlab/vl_setupnn

% Load image data
f = filesep;
load(strcat('ImageData', f, 'centCKplusimdb.mat'));

% N folds = Number of different people in the db
nfolds = 10;

addpath('Executions');
addpath('Pyramid');
addpath('bsif_code_and_data');
addpath(strcat('bsif_code_and_data', f, 'texturefilters'));

%% Execution with original CKplus db images
if (~exist('CVOcentCKplus'))
   %CVOcentCKplus = cvpartition(centCKplusimgs.labels, 'k', nfolds);
   
   x = cell(1, length(centfacesCKplus));
   
   for kk=1:length(centfacesCKplus)
      x{1,kk} = centfacesCKplus{1,kk}.id; 
   end
   
   keys = unique(x);
   
   values = cell(1, length(keys));
   for kk = 1:length(keys)
       values{kk} = kk;
   end
   
   mapObj = containers.Map(keys, values);
   
   
    
    CVOcentCKplus.NumTestSets = nfolds;
    CVOcentCKplus.training = cell(1, nfolds);
    CVOcentCKplus.test = cell(1, nfolds);
    CVOcentCKplus.TrainSize = zeros(1, nfolds);
    CVOcentCKplus.TestSize = zeros(1, nfolds);
    CVOcentCKplus.NumObservations = length(centfacesCKplus);
    
    generalI = 0;
    
    selectedPeople = logical(zeros(1, length(centfacesCKplus)));
    
    for kk=1:nfolds
        kk
        selectedPeople = logical(zeros(1, length(centfacesCKplus)));
        if (kk<3)
            lastPerson = 11;
        else
            lastPerson = 12;
        end
        
        for i=1:lastPerson
            generalI = generalI + 1;
            aux = keys{generalI};
            ispresent = cellfun(@(s) ~isempty(strfind(aux, s.id)), ...
            centfacesCKplus);
            selectedPeople = selectedPeople | ispresent;  
        end
        
        CVOcentCKplus.TestSize(kk) = sum(selectedPeople);
        CVOcentCKplus.test{kk} = selectedPeople;
        CVOcentCKplus.TrainSize(kk) = sum(~CVOcentCKplus.test{kk});
        CVOcentCKplus.training{kk} = ~CVOcentCKplus.test{kk};
    end
   
end

% HOG
myMatrixHOGcentCKplus = HOG_General(centfacesCKplus, centCKplusimgs, ...
    6, 19, 6, 25, CVOcentCKplus);

% LBP
myMatrixLBPcentCKplus = LBP_General(centfacesCKplus, centCKplusimgs, ...
    2, 6, 2, 7, CVOcentCKplus);

% Create the half and quart images and concatenate them
newcentfacesCKplus = resize_half_and_quart(centfacesCKplus, centCKplusimgs);

% LBP of concatenation of image and half image
myMatrixLBP_halfcentCKplus = LBP_half_General(centfacesCKplus, ...
    newcentfacesCKplus, centCKplusimgs, 2, 6, 2, 7, CVOcentCKplus);

% LBP of concatenation of image, half image and quarter image
myMatrixLBP_quartcentCKplus = LBP_quart_General(centfacesCKplus, ...
    newcentfacesCKplus, centCKplusimgs, 2, 6, 2, 7, CVOcentCKplus);

% Execution of LBP Pyramid
myMatrixLBPPyramidcentCKplus = LBP_of_pyramid_General(5, 16, ...
    centfacesCKplus, centCKplusimgs, 2, 6, 2, 7, CVOcentCKplus);

% BSIF
% This has to be done on 2 steps because the 3x3 filters have less bits
myMatrixBSIFcentCKplus = BSIF_General(centfacesCKplus, centCKplusimgs, ...
    3, 3, 5, 8, CVOcentCKplus);
myMatrixBSIF2centCKplus = BSIF_General(centfacesCKplus, centCKplusimgs, ...
    5, 11, 5, 12, CVOcentCKplus);

%BSIF of concatenation of image and half image
myMatrixBSIF_halfcentCKplus = BSIF_half_General(centfacesCKplus, ...
    newcentfacesCKplus, centCKplusimgs, 3, 3, 5, 8, CVOcentCKplus);
myMatrixBSIF_half2centCKplus = BSIF_half_General(centfacesCKplus, ...
    newcentfacesCKplus, centCKplusimgs, 5, 11, 5, 12, CVOcentCKplus);

%BSIF of concatenation of image, half and quart image
myMatrixBSIF_quartcentCKplus = BSIF_quart_General(centfacesCKplus, ...
    newcentfacesCKplus, centCKplusimgs, 3, 3, 5, 8, CVOcentCKplus);
myMatrixBSIF_quart2centCKplus = BSIF_quart_General(centfacesCKplus, ...
    newcentfacesCKplus, centCKplusimgs, 5, 11, 5, 12, CVOcentCKplus);

HybridMAEcentCKplus = Hybrid_LBP_HOG_BSIFT(centfacesCKplus, ...
    centCKplusimgs, myMatrixLBPcentCKplus, 2, 2, myMatrixHOGcentCKplus, ...
    6, 6, 5, 5, myMatrixBSIFcentCKplus, myMatrixBSIF2centCKplus, ...
    CVOcentCKplus);


HybridMAE_concatcentCKplus = Hybrid_LBP_HOG_BSIFT_Pyramid(centfacesCKplus, ...
    newcentfacesCKplus, centCKplusimgs, myMatrixLBPcentCKplus, ...
    myMatrixLBP_halfcentCKplus,myMatrixLBP_quartcentCKplus,... 
    myMatrixLBPPyramidcentCKplus, 2, 2, myMatrixHOGcentCKplus, 6, 6, 5, ...
    myMatrixBSIFcentCKplus, myMatrixBSIF2centCKplus, ...
    myMatrixBSIF_halfcentCKplus, myMatrixBSIF_half2centCKplus, ...
    myMatrixBSIF_quartcentCKplus, myMatrixBSIF_quart2centCKplus, ...
    CVOcentCKplus);

% Save results in a 'oridbresults.mat'
save('centCKplusresults.mat', 'CVOcentCKplus', 'myMatrixHOGcentCKplus', ...
    'myMatrixLBPcentCKplus', 'myMatrixLBP_halfcentCKplus', ...
    'myMatrixLBP_quartcentCKplus', 'myMatrixLBPPyramidcentCKplus', ...
    'myMatrixBSIFcentCKplus', 'myMatrixBSIF2centCKplus', ...
    'myMatrixBSIF_halfcentCKplus', 'yMatrixBSIF_half2centCKplus', ...
    'myMatrixBSIF_quartcentCKplus', 'myMatrixBSIF_quart2centCKplus',...
    'HybridMAEcentCKplus', 'HybridMAE_concatcentCKplus')