tic

load('/home/iarganda/workspace/FaceEmotions/ImageData/CKplusimdb.mat')

last = length( facesCKplus );
first = 1;


scale = 1;
parfor kk=first:last
    I = uint8( CKplusimgs.data( :, :, kk) );
    I = imresize( I, scale );
    [LEX, LEY, REX, REY, NX, NY] = detectEyesAndNose( I );
    Database( kk ).L_Eye_X = LEX / scale;
    Database( kk ).L_Eye_Y = LEY / scale;
    Database( kk ).R_Eye_X = REX / scale;
    Database( kk ).R_Eye_Y = REY / scale;
    Database( kk ).Nose_X = NX / scale;
    Database( kk ).Nose_Y = NY / scale;
end

scale = 1.5;
parfor kk=first:last
    if Database( kk ).L_Eye_X == 0
        I = uint8( CKplusimgs.data( :, :, kk) );
        I = imresize( I, scale );
        [LEX, LEY, REX, REY, NX, NY] = detectEyesAndNose( I );
        Database( kk ).L_Eye_X = LEX / scale;
        Database( kk ).L_Eye_Y = LEY / scale;
        Database( kk ).R_Eye_X = REX / scale;
        Database( kk ).R_Eye_Y = REY / scale;
        Database( kk ).Nose_X = NX / scale;
        Database( kk ).Nose_Y = NY / scale;
    end
end

scale = 2;
parfor kk=first:last
    if Database( kk ).L_Eye_X == 0
        I = uint8( CKplusimgs.data( :, :, kk) );
        I = imresize( I, scale );
        [LEX, LEY, REX, REY, NX, NY] = detectEyesAndNose( I );
        Database( kk ).L_Eye_X = LEX / scale;
        Database( kk ).L_Eye_Y = LEY / scale;
        Database( kk ).R_Eye_X = REX / scale;
        Database( kk ).R_Eye_Y = REY / scale;
        Database( kk ).Nose_X = NX / scale;
        Database( kk ).Nose_Y = NY / scale;
    end
end
toc

% Follow parameter from "Comparative Study of Human Age Estimation with or without
% Preclassification of Gender and Facial Expression", Dat Tien Nguyen, So Ra Cho,
% Kwang Yong Shin, Jae Won Bang, and Kang Ryoung Park, 2014.
k1 = 0.5;
k2 = 0.75;
k3 = 1.5;
finalWidth = 224;
finalHeight = 224;
l = finalWidth / (2*k1+1);

fixed = uint8( zeros(finalWidth, finalHeight, 3) );
fixed_points = [k1* l, k2*l; (k1*l+l),k2*l; finalWidth/2, finalHeight/2];
Rfixed = imref2d( size( fixed ) );

for kk=first:last
    %disp(kk);
    if Database( kk ).L_Eye_X ~= 0
        img = uint8( CKplusimgs.data( :, :, kk) );

        moving_points = [Database( kk ).R_Eye_X, Database( kk ).R_Eye_Y;...
            Database( kk ).L_Eye_X, Database( kk ).L_Eye_Y;...
            Database( kk ).Nose_X, Database( kk ).Nose_Y];

        mytform = fitgeotrans(moving_points, fixed_points, 'similarity');

        registered2 = imwarp( img, mytform,'FillValues', 255,'OutputView',Rfixed);

        imwrite( registered2, [ './CKplusAligned/' facesCKplus{kk}.filename]);

    else
        disp([ 'Unable to detect eyes and nose on image ' num2str(kk) ': ' facesCKplus{kk}.filename] );
    end
end
