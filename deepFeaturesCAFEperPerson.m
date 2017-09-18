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

%% CAFE centered using CV per person
auxfiles = strcat('ImageData', f, 'myfaces.mat');
centfacesdb = strcat('ImageData', f, 'centimdb.mat');

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


%% get VGG features
vggFace = 1;
gpu = 1;

if( vggFace )
    faces = getMatConvNetFeatures('/home/iarganda/workspace/matconvnet/data/models/vgg-face.mat', gpu, faces, images);
else
    faces = getMatConvNetFeatures('/home/iarganda/workspace/matconvnet/data/models/imagenet-vgg-f.mat', gpu, faces, images);
end


%% Aligned data
load(strcat('ImageData', f, 'centimdb.mat'));

if( vggFace )
    faces = getMatConvNetFeatures('/home/iarganda/workspace/matconvnet/data/models/vgg-face.mat', gpu, faces, centimages);
else
    faces = getMatConvNetFeatures('/home/iarganda/workspace/matconvnet/data/models/imagenet-vgg-f.mat', gpu, faces, centimages);
end

MAEDeep = SVMDeepFeatures( CVOcentCAFEperSubj, faces, centimages );

disp(strcat('MAE of deep features (CAFE aligned images):    ', sprintf('%f', MAEDeep)))

% Newline
disp(' ')
