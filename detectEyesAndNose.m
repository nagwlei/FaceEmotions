function [leftEye_X, leftEye_Y, rightEye_X, rightEye_Y, nose_X, nose_Y] = detectEyesAndNose( I )

% detect face
threshold = 4;
FaceDetector = vision.CascadeObjectDetector('FrontalFaceCART', 'MergeThreshold', threshold);
BBface=step( FaceDetector, I );
while size( BBface, 1 ) == 0 && threshold >= 0
    FaceDetector = vision.CascadeObjectDetector('FrontalFaceCART', 'MergeThreshold', threshold);
    BBface=step( FaceDetector, I );    
    threshold = threshold - 1;
end
if size( BBface, 1 ) ~= 1
    BBface = [1 1 size(I,1) size(I, 2)];
end
% store face box
extra = [BBface(1) BBface(2) 0 0];

% crop original image to face region
Iface = imcrop( I, BBface );

% Detect both eyes
EyeDetect = vision.CascadeObjectDetector('EyePairBig');
BB=step(EyeDetect,Iface);
if size( BB, 1 ) == 0
    EyeDetect = vision.CascadeObjectDetector('EyePairBig', 'MergeThreshold', 0);
    BB=step(EyeDetect,Iface);
end

if size( BB, 1 ) > 1
    threshold = 1;
    EyeDetect = vision.CascadeObjectDetector('EyePairBig', 'MergeThreshold', threshold);
    BB=step(EyeDetect,Iface);
    
    while size( BB, 1 ) > 1
        threshold = threshold+1;
        EyeDetect = vision.CascadeObjectDetector('EyePairBig', 'MergeThreshold', threshold);
        BB=step(EyeDetect,Iface);        
    end
end

if size( BB, 1 ) ~= 1
    disp( 'No clear eyes found on image at this scale' );
    rightEye_X = 0;
    rightEye_Y = 0;
    leftEye_Y = 0;
    leftEye_X = 0;
    nose_X = 0;
    nose_Y = 0;
    return
end


% To detect each eye
halfBox = BB(1) + BB(3)/2;
I2 = imcrop( Iface, [1 1 halfBox size(Iface,2)] );

% right eye
EyeDetectRight = vision.CascadeObjectDetector('RightEyeCART');
BB2=step(EyeDetectRight, I2);
% if no eye detected, try with threshold 0
if size( BB2, 1 ) == 0
    EyeDetectRight = vision.CascadeObjectDetector('RightEye','MergeThreshold', 0);
    BB2=step(EyeDetectRight, I2);
end

if size( BB2, 1 ) > 0
    maxOverlap = 0;
    bestBB2 = BB2(1, :);
    for i = 1:size(BB2,1)
        if maxOverlap < rectint( BB2(i, :), BB )
            bestBB2 = BB2(i, :);
            maxOverlap = rectint( BB2(i, :), BB );
        end
    end

    %rightEye_X = bestBB2(1) + bestBB2(3)/2;
    %rightEye_Y = bestBB2(2) + bestBB2(4)/2;
        
    % minimnum overlapping rectangle between both eyes and right eye
    l = max( bestBB2(1), BB(1) );
    r = min( bestBB2(1)+bestBB2(3), BB(1)+BB(3) );
    b = max( bestBB2(2), BB(2) );
    t = min( bestBB2(2)+bestBB2(4), BB(2)+BB(4) );
    r3 = [l b r-l t-b];
    
    rightEye_X = r3(1) + r3(3)/2 + extra(1);
    rightEye_Y = r3(2) + r3(4)/2 + extra(2);
else
    rightEye_X = BB(1) + BB(3)* 1.0/6.0 + extra(1);
    rightEye_Y = BB(2) + BB(4)* 0.5  + extra(2);
end

if size( BB2, 1 ) == 0 || maxOverlap == 0 % if still nothing
    disp( 'No clear right eye detected' );
    rightEye_X = 0;
    rightEye_Y = 0;
    leftEye_Y = 0;
    leftEye_X = 0;
    nose_X = 0;
    nose_Y = 0;
    return
end


% left eye
EyeDetectLeft = vision.CascadeObjectDetector('LeftEyeCART');
I3 = imcrop( Iface, [halfBox 1 size(Iface,1) size(Iface,2)] );

BB3=step( EyeDetectLeft, I3 );
% if no eye detected, try with threshold 0
if size( BB3, 1 ) == 0
    EyeDetectLeft = vision.CascadeObjectDetector('LeftEye','MergeThreshold', 0);
    BB3=step(EyeDetectLeft, I3);
end

if size( BB3, 1 ) > 0
    maxOverlap = 0;
    bestBB3 = BB3(1, :);
    for i = 1:size(BB3,1)
        BB3(i,:) = BB3(i,:) + [halfBox, 0, 0, 0];
        if maxOverlap < rectint( BB3(i, :), BB )
            bestBB3 = BB3(i, :);
            maxOverlap = rectint( BB3(i, :), BB );
        end
    end

    %leftEye_X = bestBB3(1) + bestBB3(3)/2;
    %leftEye_Y = bestBB3(2) + bestBB3(4)/2;
    
    % minimnum overlapping rectangle between both eyes and left eye
    l = max( bestBB3(1), BB(1) );
    r = min( bestBB3(1)+bestBB3(3), BB(1)+BB(3) );
    b = max( bestBB3(2), BB(2) );
    t = min( bestBB3(2)+bestBB3(4), BB(2)+BB(4) );
    r3 = [l b r-l t-b];
    
    leftEye_X = r3(1) + r3(3)/2 + extra(1);
    leftEye_Y = r3(2) + r3(4)/2 + extra(2);
else
    leftEye_X = BB(1) + BB(3)*5.0/6.0 + extra(1);
    leftEye_Y = BB(2) + BB(4)*0.5 + extra(2);
end
if size( BB3, 1 ) == 0 || maxOverlap == 0 % if still nothing
    disp( 'No clear left eye detected' );
    rightEye_X = 0;
    rightEye_Y = 0;
    leftEye_Y = 0;
    leftEye_X = 0;
    nose_X = 0;
    nose_Y = 0;
    return
end


% detect nose
threshold = 4;
maxThreshold = 20;
NoseDetector = vision.CascadeObjectDetector('Nose', 'MergeThreshold', threshold);
BBNose=step( NoseDetector, Iface );
while size( BBNose, 1 ) > 1 && threshold < maxThreshold
    threshold = threshold + 1;
    NoseDetector = vision.CascadeObjectDetector('Nose', 'MergeThreshold', threshold);
    BBNose=step(NoseDetector, Iface);
end
if size( BBNose, 1 ) ~= 1
    BBNose = [BBface(3)/2 BBface(4)/2 0 0];
end

nose_X = BBNose(1) + BBNose(3)/2 + extra(1);
nose_Y = BBNose(2) + BBNose(4)/2 + extra(2);

if nose_Y <= leftEye_Y && nose_Y <= rightEye_Y 
    disp( 'No clear nose found on image at this scale' );
    rightEye_X = 0;
    rightEye_Y = 0;
    leftEye_Y = 0;
    leftEye_X = 0;
    nose_X = 0;
    nose_Y = 0;
end

