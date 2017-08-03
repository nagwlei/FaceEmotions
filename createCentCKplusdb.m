f = filesep;
load(strcat('ImageData', f, 'CKplusimdb.mat'));

path = 'CKplusAligned';

centfacesCKplus = cell(1, (327*4));

centCKplusimgs.data = single(zeros(224, 224, length(facesCKplus)));

for kk=1:length(facesCKplus)
    kk
    imdir = strcat(path, f, facesCKplus{1,kk}.filename);
    image = imread(imdir);
    
    centCKplusimgs.data(:,:,kk) = single(imresize(image, [224 224]));
    centCKplusimgs.labels(1, kk) = CKplusimgs.labels(1, kk);
    centCKplusimgs.set(1, kk) = CKplusimgs.set(1, kk);
    
    % 
    centfacesCKplus{1, kk}.isOpen = facesCKplus{1, kk}.isOpen;
    centfacesCKplus{1, kk}.emotion = facesCKplus{1, kk}.emotion;
    centfacesCKplus{1, kk}.gender = facesCKplus{1, kk}.gender;
    centfacesCKplus{1, kk}.etnicity = facesCKplus{1, kk}.etnicity;

    centfacesCKplus{1, kk}.filename = facesCKplus{1, kk}.filename;
    centfacesCKplus{1, kk}.path = strcat(path, f);
    centfacesCKplus{1, kk}.id = facesCKplus{1, kk}.id;  
end

centCKplusimgs.data_mean = mean(centCKplusimgs.data, 3);

save( 'centCKplusimdb.mat', 'centfacesCKplus', 'centCKplusimgs', 'meta' );