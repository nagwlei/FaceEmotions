load('myfaces.mat');
load('Mis_ficheros\imdb.mat');

newfaces = cell(1, 1192);

for i=1:length(faces)
    %i
    img = images.data(:,:,:,i);
    half = imresize(img, 0.5);
    quart = imresize(img, 0.25);
    newfaces{i}.half = half;
    newfaces{i}.quarter = quart;
end;

LBPx = 2;
LBPy = 2;

myMatrixLBP = zeros(4, 6);
 
 for zx = 1:6
     for zy = 1:4
        % Extract LBP features
        for i=1:length(faces)
            %i;
            % Obtain selected image
            img = images.data(:,:,:,i);
            % Extract LBP features
            lbpimg = extractLBPFeatures(rgb2gray(uint8(img)),'Upright',false, 'CellSize', [16 16], 'NumNeighbors',LBPx,'Radius',LBPy);
            half = newfaces{i}.half;
            lbphalfimg = extractLBPFeatures(rgb2gray(uint8(half)),'Upright',false, 'CellSize', [16 16], 'NumNeighbors',LBPx,'Radius',LBPy);
            finallbp = horzcat(lbpimg, lbphalfimg);
            faces{i}.LBP = finallbp;
        end;
        
        % Create the folds
        CVO = cvpartition(images.labels, 'k', 5);
        %CVO = cvpartition(images.labels, 'k', 10);
        errLBP = zeros(CVO.NumTestSets, 1);

        for i = 1:CVO.NumTestSets
            TrLBP = [];
            TsLBP = [];

            trIdx = CVO.training(i);
            teIdx = CVO.test(i);
    
            for j = 1:length(faces)
                %j
                if (teIdx(j)>0)
                    %disp('IF');
                    TsLBP = vertcat(TsLBP, faces{j}.LBP);
                else
                    %disp('ELSE');
                    TrLBP = vertcat(TrLBP, faces{j}.LBP);
                end;
            end;
    
    
            Mdl = fitcecoc(TrLBP, images.labels(trIdx));
            ytestLBP = predict(Mdl, TsLBP);
    
            errLBP(i) = sum(ytestLBP~=images.labels(teIdx)');
        end;
        
        cvErrLBP = sum(errLBP)/sum(CVO.TestSize);
        
        
        myMatrixLBP(zy, zx) = cvErrLBP;
        
        
        disp(strcat('VUELTA                zx:', int2str(zx), '    zy:', int2str(zy)))
        %disp(strcat('VALORRRRRRR                :    ', sprintf('%f', cvErrHOG)))
        disp(strcat('VALOR             LBP:    ', sprintf('%f', cvErrLBP)))
        % Go to the next element of the table
        LBPx = LBPx + 2^(zy);
        disp(strcat('CellSize  after     :', int2str(LBPx)));
    end;
    LBPx = 2;
    LBPy = LBPy +1;
end;
