%% load image data existing CAFE results (with CV object)
if exist( 'oriCAFEresults.mat', 'file') == 2
    load oriCAFEresults;
end

% Load original CAFE image data
f = filesep;
load(strcat('ImageData', f, 'myfaces.mat'));
load(strcat('ImageData', f, 'imdb.mat'));

% N folds = Number of different people in the db
nfolds = 10;

%% Execution with original CAFE db images
% Create the partitions if they do not exist
if (~exist('CVO'))
    CVO = cvpartition(images.labels, 'k', nfolds);
end


%% get VGG features
vggFace = 0;
gpu = 1;

if( vggFace )
    faces = getMatConvNetFeatures('/home/iarganda/workspace/matconvnet/data/models/vgg-face.mat', gpu, faces, images);
else
    faces = getMatConvNetFeatures('/home/iarganda/workspace/matconvnet/data/models/imagenet-vgg-f.mat', gpu, faces, images);
end


%% SVM learning
MAEDeep = SVMDeepFeatures( CVO, faces, images );

disp(strcat('MAE of deep features (CAFE original images):    ', sprintf('%f', MAEDeep)))

% Newline
disp(' ')

%% Aligned data
load(strcat('ImageData', f, 'centimdb.mat'));

if( vggFace )
    faces = getMatConvNetFeatures('/home/iarganda/workspace/matconvnet/data/models/vgg-face.mat', gpu, faces, centimages);
else
    faces = getMatConvNetFeatures('/home/iarganda/workspace/matconvnet/data/models/imagenet-vgg-f.mat', gpu, faces, centimages);
end

MAEDeep = SVMDeepFeatures( CVO, faces, centimages );

disp(strcat('MAE of deep features (CAFE aligned images):    ', sprintf('%f', MAEDeep)))

% Newline
disp(' ')
