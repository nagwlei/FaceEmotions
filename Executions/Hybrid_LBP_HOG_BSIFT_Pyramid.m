function HybridMAE = Hybrid_LBP_HOG_BSIFT_Pyramid(faces, newfaces, ...
    images, resultsLBP, resultsLBP_half, resultsLBP_quart, ...
    resultsLBP_pyramid, LBPnNeighStart, LBPradiusStart, resultsHOG, ...
    HOGcellsStart, HOGnBindsStart, BSIFbitsStart, ...
    resultsBSIF, resultsBSIF2, resultsBSIF_half, resultsBSIF2_half, ...
    resultsBSIF_quart, resultsBSIF2_quart, CVO)

    LBPbest = zeros(1, 4);
    
    aux = min(min(resultsLBP));
    LBPbest(1) = aux(1);
    aux2 = min(min(resultsLBP_half));
    LBPbest(2) = aux2(1);
    aux3 = min(min(resultsLBP_quart));
    LBPbest(3) = aux3(1);
    aux4 = min(min(resultsLBP_pyramid));
    LBPbest(4) = aux4(1);

    bestTable = find(min(LBPbest));

    switch (bestTable)
        case 1
            [LBPnNeighs, LBPradius] = find(resultsLBP == LBPbest(1));
        case 2
            [LBPnNeighs, LBPradius] = find(resultsLBP_half == LBPbest(2));
        case 3
            [LBPnNeighs, LBPradius] = find(resultsLBP_quart == LBPbest(3));
        case 4
            [LBPnNeighs, LBPradius] = find(resultsLBP_pyramid == LBPbest(4));
        otherwise
            [LBPnNeighs, LBPradius] = find(resultsLBP == LBPbest(1));
    end

    minHOG = min(min(resultsHOG));
    [HOGcells, HOGnBinds] = find(resultsHOG == minHOG(1));

    BSIFbest = zeros(1,3);
    
    aux = min(min(min(resultsBSIF)),min(min(resultsBSIF2)));
    BSIFbest(1) = aux(1);
    aux2 = min(min(min(resultsBSIF_half)), min(min(resultsBSIF2_half)));
    BSIFbest(2) = aux2(1);
    aux3 = min(min(min(resultsBSIF_quart)), min(min(resultsBSIF2_quart)));
    BSIFbest(3) = aux3(1);

    bestTable = find(min(BSIFbest));
    
    switch (bestTable)
        case 1
            aux = min(min(resultsBSIF));
            minBSIF = aux(1);
            aux2 = min(min(resultsBSIF2));
            minBSIF2 = aux2(1);
            if (minBSIF(1)<minBSIF2(1))
                [BSIFfSize, BSIFbits] = find(resultsBSIF == minBSIF(1));
                %fprintf('CASE 1: IF       BSIFfSize: %d     BSIFbits: %d', BSIFfSize, BSIFbits);
            else   
                [BSIFfSize, ~] = find(resultsBSIF2 == minBSIF2(1));
                BSIFbits = 3;
                %fprintf('CASE 1: ELSE       BSIFfSize: %d     BSIFbits: %d', BSIFfSize, BSIFbits);
            end
        case 2
            aux = min(min(resultsBSIF_half));
            minBSIF = aux(1);
            aux2 = min(min(resultsBSIF2_half));
            minBSIF2 = aux2(1);
            if (minBSIF<minBSIF2)
                [BSIFfSize, BSIFbits] = find(resultsBSIF_half == minBSIF(1));
                %fprintf('CASE 1: IF       BSIFfSize: %d     BSIFbits: %d', BSIFfSize, BSIFbits);
            else
                [BSIFfSize, ~] = find(resultsBSIF2_half == minBSIF2(1));
                BSIFbits = 3;  
                %fprintf('CASE 1: ELSE       BSIFfSize: %d     BSIFbits: %d', BSIFfSize, BSIFbits);
            end
        case 3
            aux = min(min(resultsBSIF_quart));
            minBSIF = aux(1);
            aux2 = min(min(resultsBSIF2_quart));
            minBSIF2 = aux2(1);
            if (minBSIF<minBSIF2)
                [BSIFfSize, BSIFbits] = find(resultsBSIF_quart == minBSIF(1));
                %fprintf('CASE 1: IF       BSIFfSize: %d     BSIFbits: %d', BSIFfSize, BSIFbits);
            else
                [BSIFfSize, ~] = find(resultsBSIF2_quart == minBSIF2(1));
                BSIFbits = 3;
                %fprintf('CASE 1: ELSE       BSIFfSize: %d     BSIFbits: %d', BSIFfSize, BSIFbits);
            end
            
        otherwise
            aux = min(min(resultsBSIF));
            minBSIF = aux(1);
            aux2 = min(min(resultsBSIF2));
            minBSIF2 = aux2(1);
            if (minBSIF(1)<minBSIF2(1))
                [BSIFfSize, BSIFbits] = find(resultsBSIF == minBSIF(1));
                %fprintf('CASE 1: IF       BSIFfSize: %d     BSIFbits: %d', BSIFfSize, BSIFbits);
            else
                [BSIFfSize, ~] = find(resultsBSIF2 == minBSIF2(1));
                BSIFbits = 3;
                %fprintf('CASE 1: ELSE       BSIFfSize: %d     BSIFbits: %d', BSIFfSize, BSIFbits);
            end
    end
    
    LBPnNeighs = LBPnNeighs(1);
    LBPradius = LBPradius(1);
    
    HOGcells = HOGcells(1);
    HOGnBinds = HOGnBinds(1);
    
    BSIFfSize = BSIFfSize(1);
    BSIFbits = BSIFbits(1);
    
    % Go through all the images
    for i=1:length(faces)
        i
        % Obtain selected image
        if (length(size(images.data))>3)
            aux = images.data(:,:,:,i);
            img = rgb2gray(uint8(aux));
        else
            aux = images.data(:,:,i);
            img = uint8(aux);
        end
            
        % Extract best LBP features
        newLBPradius = (LBPradiusStart + LBPradius) - 1;
        %newLBPnNeighs = 2^(LBPnNeighStart + LBPnNeighs-1);
        newLBPnNeighs = 2^(LBPnNeighs);
        
        % We sppose it always starts in 2 LBPnNeighStart
        LBPfeatures = extractLBPFeatures(img, 'Upright',false, ...
            'CellSize', [16 16], 'NumNeighbors', ...
            newLBPnNeighs,'Radius',newLBPradius);
        
        % Extract best HOG features
        newHOGcells = (HOGcells + HOGcellsStart) - 1;
        newHOGnBinds = (HOGnBinds + HOGnBindsStart) - 1;
         
        [featureVector,~] = ...
            extractHOGFeatures(img, 'CellSize', [newHOGcells newHOGcells], ...
                    'NumBins', newHOGnBinds);
        
        % Extract best BSIF features
        % Start is always 3
        newBSIFfSize = 3 + (2 * (BSIFfSize - 1));
        newBSIFbits = (BSIFbits + BSIFbitsStart) - 1;
        
        f = filesep;
        ICAtextureFiltersdir = strcat('bsif_code_and_data', f, ...
            'texturefilters', f, 'ICAtextureFilters_', ...
            num2str(newBSIFfSize), 'x', num2str(newBSIFfSize), '_', ...
            num2str(newBSIFbits), 'bit');
                
        % normalized BSIF code word histogram
        load(ICAtextureFiltersdir);
        BSIFfeatures = bsif(double(img), ICAtextureFilters,'nh');
        
        % Save concatenation of featuresof the different classifiers
        faces{i}.Hybrid = horzcat(LBPfeatures, featureVector, BSIFfeatures);
    end;
            
    errHybrid = zeros(CVO.NumTestSets, 1);

    for i = 1:CVO.NumTestSets
        trIdx = CVO.training(i);
        teIdx = CVO.test(i);
        
         % This is necessary to work with the CVO created for JAFFE
        if (strcmp(class(trIdx), 'cell'))
           trIdx = trIdx{1}; 
        end
        if (strcmp(class(teIdx), 'cell'))
            teIdx = teIdx{1};
        end

        TrHybrid = zeros(sum(trIdx), length(faces{1}.Hybrid));
        TeHybrid = zeros(sum(teIdx), length(faces{1}.Hybrid));

        tr = 0;
        te = 0;

        for j = 1:length(faces)
            if (teIdx(j)>0)
                te = te + 1;
                TeHybrid(te,:) = faces{j}.Hybrid;
            else
                tr = tr + 1;
                TrHybrid(tr,:) = faces{j}.Hybrid;
            end;
        end;

        t = templateSVM( 'Standardize', 1 );
        Mdl = fitcecoc(TrHybrid, images.labels(trIdx), 'Learners', t);
        ytestHybrid = predict(Mdl, TeHybrid);

        errHybrid(i) = sum(ytestHybrid~=images.labels(teIdx)');
    end;

   % MAE value for the Hybrid classifier
   HybridMAE = sum(errHybrid)/sum(CVO.TestSize);

   disp(strcat('MAE of       Hybrid:    ', sprintf('%f', HybridMAE)))
end