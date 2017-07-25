function HybridMAE = Hybrid_LBP_HOG_BSIFT_Pyramid(faces, newfaces, ...
    images, resultsLBP, resultsLBP_half, resultsLBP_quart, ...
    resultsLBP_pyramid, LBPnNeighStart, LBPradiusStart, resultsHOG, ...
    HOGcellsStart, HOGnBindsStart, BSIFbitsStart, ...
    resultsBSIF, resultsBSIF2, resultsBSIF_half, resultsBSIF2_half, ...
    resultsBSIF_quart, resultsBSIF2_quart, CVO)

    LBPbest = zeros(1, 4);
    
    LBPbest(1) = unique(min(min(resultsLBP)));
    LBPbest(2) = unique(min(min(resultsLBP_half)));
    LBPbest(3) = unique(min(min(resultsLBP_quart)));
    LBPbest(4) = unique(min(min(resultsLBP_pyramid)));

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
    
    BSIFbest(1) = unique(min(min(min(resultsBSIF)),min(min(resultsBSIF2))));
    BSIFbest(2) = unique(min(min(min(resultsBSIF_half)), min(min(resultsBSIF2_half))));
    BSIFbest(3) = unique(min(min(min(resultsBSIF_quart)), min(min(resultsBSIF2_quart))));

    bestTable = find(min(BSIFbest));
    
    switch (bestTable)
        case 1
            minBSIF = unique(min(min(resultsBSIF)));
            minBSIF2 = unique(min(min(resultsBSIF2)));
            if (minBSIF(1)<minBSIF2(1))
                [BSIFfSize, BSIFbits] = find(resultsBSIF == minBSIF(1));
            else
                BSIFfSize = 3;
                BSIFbits = find(resultsBSIF2 == minBSIF2(1));
            end
        case 2
            minBSIF = unique(min(min(resultsBSIF_half)));
            minBSIF2 = unique(min(min(resultsBSIF2_half)));
            if (minBSIF<minBSIF2)
                [BSIFfSize, BSIFbits] = find(resultsBSIF_half == minBSIF(1));
            else
                BSIFfSize = 3;
                BSIFbits = find(resultsBSIF2_half == minBSIF2(1));
            end
        case 3
            minBSIF = unique(min(min(resultsBSIF_quart)));
            minBSIF2 = unique(min(min(resultsBSIF2_quart)));
            if (minBSIF<minBSIF2)
                [BSIFfSize, BSIFbits] = find(resultsBSIF_quart == minBSIF(1));
            else
                BSIFfSize = 3;
                BSIFbits = find(resultsBSIF2_quart == minBSIF2(1));
            end
            
        otherwise
            minBSIF = unique(min(min(resultsBSIF)));
            minBSIF2 = unique(min(min(resultsBSIF2)));
            if (minBSIF(1)<minBSIF2(1))
                [BSIFfSize, BSIFbits] = find(resultsBSIF == minBSIF(1));
            else
                BSIFfSize = 3;
                BSIFbits = find(resultsBSIF2 == minBSIF2(1));
            end
    end
    
    LBPnNeighs = unique(LBPnNeighs);
    LBPradius = unique(LBPradius);
    
    HOGcells = unique(HOGcells);
    HOGnBinds = unique(HOGnBinds);
    
    BSIFfSize = unique(BSIFfSize);
    BSIFbits = unique(BSIFbits);
    
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