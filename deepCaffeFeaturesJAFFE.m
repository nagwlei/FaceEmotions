%% load image data existing JAFFE results (with CV object)
if exist( './Results/oriJAFFEresults.mat', 'file') == 2
    load ./Results/oriJAFFEresults;
end

% Load image data
f = filesep;
load(strcat('ImageData', f, 'jaffeimdb.mat'));

% N folds = Number of different people in the db
nfolds = 10;

%% Execution with original Jaffe db images
if (~exist('CVOjaffe', 'var'))
    mapObj = containers.Map({'KA', 'KL', 'KM', 'KR', 'MK', 'NA','NM', ...
        'TM', 'UY', 'YM'}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    CVOjaffe.NumTestSets = nfolds;
    CVOjaffe.training = cell(1, nfolds);
    CVOjaffe.test = cell(1, nfolds);
    CVOjaffe.TrainSize = zeros(1, nfolds);
    CVOjaffe.TestSize = zeros(1, nfolds);
    CVOjaffe.NumObservations = length(facesjaffe);
    
    theIds = keys(mapObj);

    classesIndexes = zeros(1, length(facesjaffe));

    for kk=1:nfolds
        disp( kk );
        aux = theIds(kk);
        ispresent = cellfun(@(s) ~isempty(strfind(aux{1}, s.id)), facesjaffe);

        %foldIndexes{1,kk} = ispresent;
        CVOjaffe.TestSize(kk) = sum(ispresent);
        CVOjaffe.test{kk} = ispresent;
        CVOjaffe.TrainSize(kk) = sum(~CVOjaffe.test{kk});
        CVOjaffe.training{kk} = ~CVOjaffe.test{kk};
    end;
end

%% get VGG features

gpu = 1;
facesjaffe = getMatCaffeFeatures('/home/iarganda/workspace/caffe/models/EmotiW/','deploy.txt','EmotiW_VGG_S.caffemodel','/home/iarganda/workspace/caffe/models/EmotiW/mean.binaryproto', gpu, facesjaffe, jaffeimgs);


%% SVM learning
MAEDeep = SVMDeepFeatures( CVOjaffe, facesjaffe, jaffeimgs );
disp(strcat('MAE of deep Caffe features (original JAFFE images):    ', sprintf('%f', MAEDeep)))

% Newline
disp(' ')
