centimages.data = single(zeros(224, 224, 3, length(faces)));

dir = '../IMAGENES_CENTRADAS/crop2/';

for kk=1:length(faces)
    kk
    im = imread(strcat(dir, faces{kk}.filename));
    centimages.data(:,:,:,kk) = single(im);
end;

centimages.data_mean = mean( centimages.data, 4 );

classes = {'angry', 'happy', 'surprised', 'neutral', 'disgust', 'sad', 'fearful' };

% clean up labels
for kk=1:length( faces )
    for jj=1:length( classes )
        if strncmpi( faces{kk}.emotion, classes{jj}, length( classes{ jj } ) )
            faces{kk}.emotion = classes{ jj };
        end
    end
end
% print wrong labels
for kk=1:length( centimages )
    if isempty( strmatch( faces{kk}.emotion, classes ) )
        kk
    end
end

% assign labels
centimages.labels = zeros(1, length( faces ) );
for kk=1:length( faces )
    centimages.labels(kk) = strmatch( faces{kk}.emotion, classes );
end


% assign set
[trainInd,valInd,testInd] = dividerand(length(faces),0.6,0.2,0.2);
centimages.set = zeros(1, length( faces ) );

centimages.set( trainInd ) = 1;
centimages.set( valInd ) = 2;
centimages.set( testInd ) = 3;

% meta info
meta.classes = classes;
meta.sets = {'train', 'val', 'test'};

save( 'imdb.mat', 'images', 'meta' );

