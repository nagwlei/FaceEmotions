% Load image data
f = filesep;

% Addresses of the images
auxfiles = strcat('ImageData', f, 'myfaces.mat');
facesdb = strcat('ImageData', f, 'imdb.mat');
centfacesdb = strcat('ImageData', f, 'centimdb.mat');
jaffedb = strcat('ImageData', f, 'jaffeimdb.mat');
centckplusdb = strcat('ImageData', f, 'centCKplusimdb.mat');

% Adresses of the Results
oricafferesults = strcat('Results', f, 'oriCAFFEresults.mat');
centcafferesults = strcat('Results', f, 'centCAFFEresults.mat');
jafferesults = strcat('Results', f, 'oriJAFFEresults.mat');
centckplusresults = strcat('Results', f, 'centCKplusresults.mat');

% N folds = Number of different people in the db
nfolds = 10;

addpath('Executions');
addpath('Executions_v2');
addpath('Pyramid');
addpath('bsif_code_and_data');
addpath(strcat('bsif_code_and_data', f, 'texturefilters'));

%% New values for CAFE
load(auxfiles);
load(facesdb);
load(oricafferesults);

% LBP values changing cellSize from 10 to 25
extraArrayLBPfacesCell10_25 = LBP_General_v2(faces, images, 10, 25, 8, 1, ...
    CVO);
newfaces = resize_half_and_quart(faces, images);
extraArrayLBP_halffacesCell10_25 = LBP_half_General_v2(faces, newfaces, ...
    images, 10, 25, 8, 1, CVO);
extraArrayLBP_quartfacesCell10_25 = LBP_quart_General_v2(faces, newfaces, ...
    images, 10, 25, 8, 1, CVO);

% HOG values
extraMatrixHOGfaces20_25 = HOG_General(faces, images, 20, 25, 6, 25, ...
    CVO);

save('extraResultsCAFE.mat', 'extraArrayLBPfacesCell10_25', ...
    'extraArrayLBP_halffacesCell10_25', 'extraArrayLBP_quartfacesCell10_25', ...
    'extraMatrixHOGfaces20_25');


%% New values for centered CAFE
clearvars -except f facesdb centfacesdb auxfiles jaffedb centckplusdb ...
    oricafferesults centcafferesults jafferesults centckplusresults ...
    nfolds faces

load(centfacesdb);
load(centcafferesults);

% LBP values changing cellSize from 10 to 25
extraArrayLBPcentfacesCell10_25 = LBP_General_v2(faces, centimages, ...
    10, 25, 8, 1, CVO);
newfacescent = resize_half_and_quart(faces, centimages);
extraArrayLBP_halfcentfacesCell10_25 = LBP_half_General_v2(faces, newfacescent, ...
    centimages, 10, 25, 8, 1, CVO);
extraArrayLBP_quartcentfacesCell10_25 = LBP_quart_General_v2(faces, newfacescent, ...
    centimages, 10, 25, 8, 1, CVO);

% HOG values
extraMatrixHOGcentfaces20_25 = HOG_General(faces, centimages, ...
    20, 25, 6, 25, CVO);

save('extraResultscentCAFE.mat', 'extraArrayLBPcentfacesCell10_25', ...
    'extraArrayLBP_halfcentfacesCell10_25', 'extraArrayLBP_quartcentfacesCell10_25', ...
    'extraMatrixHOGcentfaces20_25');

%% New values for JAFFE
clearvars -except f facesdb centfacesdb auxfiles jaffedb centckplusdb ...
    oricafferesults centcafferesults jafferesults centckplusresults ...
    nfolds faces

load(jaffedb);
load(jafferesults);

% LBP values changing cellSize from 10 to 25
extraArrayLBPjaffeCell10_25 = LBP_General_v2(facesjaffe, jaffeimgs, 10, 25, 8, 1, ...
    CVOjaffe);
newfacesjaffe = resize_half_and_quart(facesjaffe, jaffeimgs);
extraArrayLBP_halfjaffeCell10_25 = LBP_half_General_v2(facesjaffe, ...
    newfacesjaffe, jaffeimgs, 10, 25, 8, 1, CVOjaffe);
extraArrayLBP_quartjaffeCell10_25 = LBP_quart_General_v2(facesjaffe, ...
    newfacesjaffe, jaffeimgs, 10, 25, 8, 1, CVOjaffe);

% HOG values
extraMatrixHOGjaffe20_25 = HOG_General(facesjaffe, jaffeimgs, 20, 25, 6, 25, ...
    CVOjaffe);

% Missing values BSIF of concatenation of image, half and quart image
myMatrixBSIF_quartjaffe = BSIF_quart_General(facesjaffe, newfacesjaffe, ...
    jaffeimgs, 3, 3, 5, 8, CVOjaffe);
myMatrixBSIF_quart2jaffe = BSIF_quart_General(facesjaffe, newfacesjaffe, ...
    jaffeimgs, 5, 11, 5, 12, CVOjaffe);

save('extraResultsJAFFE.mat', 'extraArrayLBPjaffeCell10_25', ...
    'extraArrayLBP_halfjaffeCell10_25', ...
    'extraArrayLBP_quartjaffeCell10_25', 'extraMatrixHOGjaffe20_25', ...
    'myMatrixBSIF_quartjaffe', 'myMatrixBSIF_quart2jaffe');

%% New values for CK+
clearvars -except f facesdb centfacesdb auxfiles jaffedb centckplusdb ...
    oricafferesults centcafferesults jafferesults centckplusresults ...
    nfolds faces

load(centckplusdb);
load(centckplusresults);

% LBP values changing cellSize from 10 to 25
extraArrayLBPcentCKplusCell10_25 = LBP_General_v2(centfacesCKplus, centCKplusimgs, ...
    10, 25, 8, 1, CVOcentCKplus);
newcentfacesCKplus = resize_half_and_quart(centfacesCKplus, centCKplusimgs);
extraArrayLBP_halfcentCKplusCell10_25 = LBP_half_General_v2(centfacesCKplus, ...
    newcentfacesCKplus, centCKplusimgs, 10, 25, 8, 1, CVOcentCKplus);
extraArrayLBP_quartcentCKplusCell10_25 = LBP_half_General_v2(centfacesCKplus, ...
    newcentfacesCKplus, centCKplusimgs, 10, 25, 8, 1, CVOcentCKplus);

% HOG values
extraMatrixHOGcentCKplus20_25 = HOG_General(centfacesCKplus, centCKplusimgs, ...
    20, 25, 6, 25, CVOcentCKplus);

save('extraResultscentCKplus.mat', 'extraArrayLBPcentCKplusCell10_25', ...
    'extraArrayLBP_halfcentCKplusCell10_25', ...
    'extraArrayLBP_quartcentCKplusCell10_25', ...
    'extraMatrixHOGcentCKplus20_25');

%% CAFE centered using CV per person
clearvars -except f auxfiles centfacesdb centcafferesults nfolds faces

load(auxfiles);
load(centfacesdb);

% Create CV per person (There are 157 people, en each fold 15 or 16 people)
if (~exist('CVOcentCAFEperSubj'))
    
    x = cell(1, length(faces));
   
    for kk=1:length(faces)
        kk
        x{1,kk} = faces{1,kk}.id; 
    end
   
    keys = unique(x);
   
    values = cell(1, length(keys));
    for kk = 1:length(keys)
        kk
        values{kk} = kk;
    end
    
    mapObj = containers.Map(keys, values);
    
    CVOcentCAFEperSubj.NumTestSets = nfolds;
    CVOcentCAFEperSubj.training = cell(1, nfolds);
    CVOcentCAFEperSubj.test = cell(1, nfolds);
    CVOcentCAFEperSubj.TrainSize = zeros(1, nfolds);
    CVOcentCAFEperSubj.TestSize = zeros(1, nfolds);
    CVOcentCAFEperSubj.NumObservations = length(faces);        
    
    generalI = 0;
    
    selectedPeople = logical(zeros(1, length(faces)));
    
    for kk=1:nfolds
        kk
        selectedPeople = logical(zeros(1, length(faces)));
        if (kk > 7 )
            lastPerson = 16;
        else
            lastPerson = 15;
        end
        
        for i=1:lastPerson
            generalI = generalI + 1;
            aux = keys{generalI};
            ispresent = cellfun(@(s) ~isempty(strfind(aux, s.id)), ...
                faces);
            selectedPeople = selectedPeople | ispresent;
        end
        
        CVOcentCAFEperSubj.TestSize(kk) = sum(selectedPeople);
        CVOcentCAFEperSubj.test{kk} = selectedPeople;
        CVOcentCAFEperSubj.TrainSize(kk) = sum(~CVOcentCAFEperSubj.test{kk});
        CVOcentCAFEperSubj.training{kk} = ~CVOcentCAFEperSubj.test{kk};
    end
end

%HOG
myMatrixHOGcent = HOG_General(faces, centimages, 6, 25, 6, 25, CVOcentCAFEperSubj);

%LBP
myMatrixLBPcent = LBP_General(faces, centimages, 2, 6, 2, 7, CVOcentCAFEperSubj);

% Create the half and quart images and concatenate them
newfacescent = resize_half_and_quart(faces, centimages);

% LBP of concatenation of image and half image
myMatrixLBPcent_half = LBP_half_General(faces, newfacescent, ...
    centimages, 2, 6, 2, 7, CVOcentCAFEperSubj);

% LBP of concatenation of image, half image and quarter image
myMatrixLBPcent_quart = LBP_quart_General(faces, newfacescent, ...
    centimages, 2, 6, 2, 7, CVOcentCAFEperSubj);

% LBP of pyramid
myMatrixLBPcentPyramid = LBP_of_pyramid_General(5, 16, faces, ...
    centimages, 2, 6, 2, 7, CVOcentCAFEperSubj);

% BSIF
% This has to be done on 2 steps because the 3x3 filters have less bits
myMatrixBSIFcent = BSIF_General(faces, centimages, 3, 3, 5, 8, CVOcentCAFEperSubj);
myMatrixBSIFcent2 = BSIF_General(faces, centimages, 5, 11, 5, 12, CVOcentCAFEperSubj);

%BSIF of concatenation of image and half image
myMatrixBSIFcent_half = BSIF_half_General(faces, newfacescent, ...
    centimages, 3, 3, 5, 8, CVOcentCAFEperSubj);
myMatrixBSIFcent_half2 = BSIF_half_General(faces, newfacescent, ...
    centimages, 5, 11, 5, 12, CVOcentCAFEperSubj);

%BSIF of concatenation of image, half and quart image
myMatrixBSIFcent_quart = BSIF_quart_General(faces, newfacescent, ...
    centimages, 3, 3, 5, 8, CVOcentCAFEperSubj);
myMatrixBSIFcent_quart2 = BSIF_quart_General(faces, newfacescent, ...
    centimages, 5, 11, 5, 12, CVOcentCAFEperSubj);

% Hybrid classifier
HybridMAEcent = Hybrid_LBP_HOG_BSIFT(faces, centimages, myMatrixLBPcent, ...
    2, 2, myMatrixHOGcent, 6, 6, 5, 5, myMatrixBSIFcent, ...
    myMatrixBSIFcent2, CVOcentCAFEperSubj);

HybridMAEcent_concat = Hybrid_LBP_HOG_BSIFT_Pyramid(faces, newfacescent, ...
    centimages, myMatrixLBPcent, myMatrixLBPcent_half, myMatrixLBPcent_quart,... 
    myMatrixLBPcentPyramid, 2, 2, myMatrixHOGcent, 6, 6, 5, ...
    myMatrixBSIFcent, myMatrixBSIFcent2, myMatrixBSIFcent_half, ...
    myMatrixBSIFcent_half2, myMatrixBSIFcent_quart, ...
    myMatrixBSIFcent_quart2, CVOcentCAFEperSubj);

% LBP values changing cellSize from 10 to 25 (with fixed Radius and
% Nneighs)
% LBP values changing cellSize from 10 to 25
f = filesep;
oricafferesults = strcat('Results', f, 'oriCAFFEresults.mat');
centfacesdb = strcat('ImageData', f, 'centimdb.mat');

load(oricafferesults);
load(centfacesdb);

newfacescent = resize_half_and_quart(faces, centimages);

extraArrayLBPCell10_25 = LBP_General_v2(faces, centimages, ...
    10, 25, 8, 1, CVO);
extraArrayLBP_halfCell10_25 = LBP_half_General_v2(faces, newfacescent, ...
    centimages, 10, 25, 8, 1, CVO);
extraArrayLBP_quartCell10_25 = LBP_quart_General_v2(faces, newfacescent, ...
    centimages, 10, 25, 8, 1, CVO);

save('extraCAFEbyPersonCV.mat', 'myMatrixHOGcent', 'myMatrixLBPcent', ...
    'myMatrixLBPcent_half', 'myMatrixLBPcent_quart', ...
    'myMatrixLBPcentPyramid', 'myMatrixBSIFcent', 'myMatrixBSIFcent2', ...
    'myMatrixBSIFcent_half', 'myMatrixBSIFcent_half2', ...
    'myMatrixBSIFcent_quart', 'myMatrixBSIFcent_quart2', 'HybridMAEcent', ...
    'HybridMAEcent_concat', 'extraArrayLBPCell10_25', ...
    'extraArrayLBP_halfCell10_25', 'extraArrayLBP_quartCell10_25');
