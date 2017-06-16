load('myfaces.mat');
load('Mis_ficheros\imdb.mat');

%for i = 1:length(faces)
    %i;
    % In the 4D images the 4th dimension is the image
    %img = images.data(:,:,:,i);
%end;

%LBPx = 4;
%LBPy = 2;
HOGx = 6;
%HOGy = 6;
HOGy = 20;
%myMatrixLBP = zeros(17, 9);
myMatrixHOG = zeros(7, 14);

for zx = 1:7
    for zy = 1:14 
        % Extract HOG and SIFT features
        for i=1:length(faces)
            %i;
            % Obtain selected image
            img = images.data(:,:,:,i);
            % Extract LBP features
            %lbpimg = extractLBPFeatures(rgb2gray(img),'Upright',false,'NumNeighbors',LBPx,'Radius',LBPy);
            %faces{i}.LBP = lbpimg;
            % Extract HOG features
            [featureVector,hogVisualization] = extractHOGFeatures(img, 'CellSize', [HOGx HOGx], 'NumBins', HOGy);
            faces{i}.HOG = featureVector;
        end;
        
        % Create the folds
        CVO = cvpartition(images.labels, 'k', 5);
        %CVO = cvpartition(images.labels, 'k', 10);
        %CVO = cvpartition(images.set, 'k', 7);
        %errLBP = zeros(CVO.NumTestSets, 1);
        errHOG = zeros(CVO.NumTestSets, 1);

        for i = 1:CVO.NumTestSets
            %TrLBP = [];
            %TsLBP = [];
            TrHOG = [];
            TsHOG = [];
            trIdx = CVO.training(i);
            teIdx = CVO.test(i);
    
            for j = 1:length(faces)
                %j
                if (teIdx(j)>0)
                    %disp('IF');
                    %TsLBP = vertcat(TsLBP, faces{j}.LBP);
                    TsHOG = vertcat(TsHOG, faces{j}.HOG);
                else
                    %disp('ELSE');
                    %TrLBP = vertcat(TrLBP, faces{j}.LBP);
                    TrHOG = vertcat(TrHOG, faces{j}.HOG);
                end;
            end;
    
    
            %Mdl = fitcecoc(TrLBP, images.labels(trIdx));
            %ytestLBP = predict(Mdl, TsLBP);
    
            Mdl2 = fitcecoc(TrHOG, images.labels(trIdx));
            ytestHOG = predict(Mdl2, TsHOG);
    
            %errLBP(i) = sum(ytestLBP~=images.labels(teIdx)');
            errHOG(i) = sum(ytestHOG~=images.labels(teIdx)');

        end;
        %cvErrLBP = sum(errLBP)/sum(CVO.TestSize);
        cvErrHOG = sum(errHOG)/sum(CVO.TestSize);
        
        %myMatrixLBP(zx, zy) = cvErrLBP;
        myMatrixHOG(zx, zy) = cvErrHOG; 
        
        disp(strcat('VUELTAAAAAAA                zx:', int2str(zx), '    zy:', int2str(zy)))
        disp(strcat('VALORRRRRRR                :    ', sprintf('%f', cvErrHOG)))
        %disp(strcat('VALORRRRRRR             LBP:    ', sprintf('%f', cvErrLBP)))
        % Go to the next element of the table
        %LBPx = LBPx + 1;
        HOGx = HOGx + 1;
    end;
    %LBPx = 4;
    HOGx = 6;
    %LBPy = LBPy +1;
    HOGy = HOGy + 1;
end;





