%% load image data existing CKplus results (with CV object)
if exist( 'centCKplusresults.mat', 'file') == 2
    load centCKplusresults;
end

% Load image data
f = filesep;
load(strcat('ImageData', f, 'centCKplusimdb.mat'));

% N folds = Number of different people in the db
nfolds = 10;

%% Create CV per person
if (~exist('CVOcentCKplus', 'var'))
   
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
        %kk
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


%% get VGG features
vggFace = 0;
gpu = 1;

if( vggFace )
    centfacesCKplus = getMatConvNetFeatures('/home/iarganda/workspace/matconvnet/data/models/vgg-face.mat', gpu, centfacesCKplus, centCKplusimgs);
else
    centfacesCKplus = getMatConvNetFeatures('/home/iarganda/workspace/matconvnet/data/models/imagenet-vgg-f.mat', gpu, centfacesCKplus, centCKplusimgs);
end


%% SVM learning
MAEDeep = SVMDeepFeatures( CVOcentCKplus, centfacesCKplus, centCKplusimgs );

disp(strcat('MAE of deep features (CK+ aligned images):    ', sprintf('%f', MAEDeep)))

% Newline
disp(' ')

