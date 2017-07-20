% Address where the images are
directory = 'jaffe/';

% Get image names in the directory
imagefiles = dir(strcat(directory, '*.tiff'));  

% Structure to save the data
jaffeimgs.data = single(zeros(224, 224, length(imagefiles)));

for kk=1:length(imagefiles)
    kk
    im = imread(strcat(directory, imagefiles(kk).name));
    jaffeimgs.data(:,:,kk) = single(imresize(im, [224 224]));
end;

jaffeimgs.data_mean = mean( jaffeimgs.data, 3 );

% Create mapping with the abbreviature of the emotion in the images and the
% real emotion
classes = {'angry', 'happy', 'surprised', 'neutral', 'disgust', 'sad', 'fearful' };
mapObj = containers.Map({'AN', 'HA', 'SU', 'NE', 'DI', 'SA', 'FE'}, classes);

facesjaffe = cell(1,length( jaffeimgs ));

% clean up labels and create new ones with the same structure as imdb
for kk=1:length( imagefiles )
    kk
    facesjaffe{kk}.isopen = '';
    facesjaffe{kk}.emotion = mapObj(imagefiles(kk).name(4:5));
    facesjaffe{kk}.gender = 'F';
    facesjaffe{kk}.etnicity = 'AS';
    facesjaffe{kk}.filename = imagefiles(kk).name;
    path = directory;
    facesjaffe{kk}.id = imagefiles(kk).name(1:2);
end

% assign labels
jaffeimgs.labels = zeros(1, length( imagefiles ) );
for kk=1:length( imagefiles )
    jaffeimgs.labels(kk) = strmatch( facesjaffe{kk}.emotion, classes );
end


% assign set
[trainInd,valInd,testInd] = dividerand(length(facesjaffe),0.6,0.2,0.2);
jaffeimgs.set = zeros(1, length( imagefiles ) );

jaffeimgs.set( trainInd ) = 1;
jaffeimgs.set( valInd ) = 2;
jaffeimgs.set( testInd ) = 3;

% meta info
meta.classes = classes;
meta.sets = {'train', 'val', 'test'};

save( 'jaffeimdb.mat', 'jaffeimgs', 'facesjaffe', 'meta' );