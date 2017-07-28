f = filesep;

% Address where the images are
imDir = 'CK+\cohn-kanade-images';

% Address where the emotion labels are
emoDir = 'CK+\Emotion';

d = dir(imDir);
isub = [d(:).isdir];
nameFolds = {d(isub).name}';
nameFolds(ismember(nameFolds, {'.', '..'})) = [];

facesCKplus = cell(1, (327*4));
i = 1;

classesInCKplus = {'neutral', 'angry', 'contempt', 'disgust', ...
    'fearful', 'happy', 'sad', 'surprised'};

%Inside the folders inside CK+ folder
for kk=1:length(nameFolds)
    kk
    subDir = strcat(emoDir, f, nameFolds{kk});
    d2 = dir(subDir);
    isub2 = [d2(:).isdir];
    nameFolds2 = {d2(isub2).name}';
    nameFolds2(ismember(nameFolds2, {'.', '..'})) = [];
    
    % Inside each folder check for the folder of images
    for yy=1:length(nameFolds2)
        yy
        subSubDir = strcat(subDir, f, nameFolds2{yy}, f, '*.txt');
        d3 = dir(subSubDir);
        
        % If *.txt exists it means it has a labelled emotions so the image
        % is included into the structure
        if (length({d3.name}')>0)
            % Read the emotion
            emotiontxt = strcat(subDir, f, nameFolds2{yy}, f, {d3.name}');
            
            % Get name of images in subDirectory
            subImages = strcat(imDir, f, nameFolds{kk}, f, ...
                nameFolds2{yy}, f, '*.png');
            d3img = dir(subImages);
            images = {d3img.name}';
            
            fileID = fopen(emotiontxt{1}, 'r');
            formatSpec = '%d e+00';
            emotion = fscanf(fileID, formatSpec);
            fclose(fileID);
            
            % Antepenultimate frame
            facesCKplus{1, i}.isOpen = '';
            facesCKplus{1, i}.emotion = classesInCKplus{(emotion+1)};
            facesCKplus{1, i}.gender = '';
            facesCKplus{1, i}.etnicity = '';
            
            facesCKplus{1, i}.filename = images{length(images)-2};
            facesCKplus{1, i}.path = strcat(imDir, f, nameFolds{kk}, f, nameFolds2{yy});
            facesCKplus{1, i}.id = nameFolds{kk};
            i = i + 1;
            
            % Penultimate frame
            facesCKplus{1, i}.isOpen = '';
            facesCKplus{1, i}.emotion = classesInCKplus{(emotion+1)};
            facesCKplus{1, i}.gender = '';
            facesCKplus{1, i}.etnicity = '';
            
            facesCKplus{1, i}.filename = images{length(images)-1};
            facesCKplus{1, i}.path = strcat(imDir, f, nameFolds{kk}, f, nameFolds2{yy});
            facesCKplus{1, i}.id = nameFolds{kk};
            i = i + 1;
            
            % Last frame
            facesCKplus{1, i}.isOpen = '';
            facesCKplus{1, i}.emotion = classesInCKplus{(emotion+1)};
            facesCKplus{1, i}.gender = '';
            facesCKplus{1, i}.etnicity = '';

            facesCKplus{1, i}.filename = images{length(images)};
            facesCKplus{1, i}.path = strcat(imDir, f, nameFolds{kk}, f, nameFolds2{yy});
            facesCKplus{1, i}.id = nameFolds{kk};
            i = i + 1;            
            
            
            % NEUTRAL FACE
            facesCKplus{1, i}.isOpen = '';
            facesCKplus{1, i}.emotion = 'neutral';
            facesCKplus{1, i}.gender = '';
            facesCKplus{1, i}.etnicity = '';
            facesCKplus{1, i}.filename = images{1};
            facesCKplus{1, i}.path = strcat(imDir, f, nameFolds{kk}, f, nameFolds2{yy});
            facesCKplus{1, i}.id = nameFolds{kk};
            i = i + 1;
        end   
    end 
end

CKplusimgs.data = single(zeros(224, 224, length(facesCKplus)));

for kk=1:length(facesCKplus)
    kk
    im = imread(strcat(facesCKplus{1, kk}.path, f, facesCKplus{1,kk}.filename));
    
    % To have all the database in gray
    [~, ~, auxrgb] = size(im);
    if (auxrgb>2)
       im =  rgb2gray(im);
    end
    CKplusimgs.data(:,:,kk) = single(imresize(im, [224 224]));
end

CKplusimgs.data_mean = mean(CKplusimgs.data, 3);

% Assign emotion labels
classes = {'angry', 'happy', 'surprised', 'neutral', 'disgust', 'sad', 'fearful', 'contempt'};
CKplusimgs.labels = zeros(1, length(facesCKplus));
for kk=1:length(facesCKplus)
    kk
    CKplusimgs.labels(1, kk) = strmatch(facesCKplus{kk}.emotion, classes);
end

% Asign set
[trainInd, valInd, testInd] = dividerand(length(facesCKplus), 0.6, 0.2, 0.2);
CKplusimgs.set = zeros(1, length(facesCKplus));

CKplusimgs.set( trainInd )=1;
CKplusimgs.set( valInd ) = 2;
CKplusimgs.set( testInd ) = 3;

% Meta info
meta.classes = classes;
meta.sets = {'train', 'val', 'test'};

save('CKplusimdb.mat', 'CKplusimgs', 'facesCKplus', 'meta');