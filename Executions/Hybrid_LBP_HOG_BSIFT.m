

function HybridMAE = Hybrid_LBP_HOG_BSIFT(faces, images, resultsLBP, ...
    LBPnNeighStart, LBPradiusStart, resultsHOG, HOGcellsStart, HOGnBindsStart, ...
    BSIFfSizeStart,BSIFbitsStart, resultsBSIF, resultsBSIF2, CVO)

    minLBP = min(min(resultsLBP));
    [LBPnNeighs, LBPradius] = find(resultsLBP == minLBP(1));
    minHOG = min(min(resultsHOG));
    [HOGcells, HOGnBinds] = find(resultsHOG == minHOG(1));
    minBSIF = min(min(resultsBSIF));
    minBSIF2 = min(resultsBSIF2);
    if (minBSIF<minBSIF2)
        [BSIFfSize, BSIFbits] = find(resultsBSIF == minBSIF(1));
    else
        BSIFfSize = 3;
        BSIFbits = find(resultsBSIF2 == minBSIF2(1));
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
        
        % We sppose it always starts in 2 LBPnNeighStart
        %newLBPnNeighs = 2^(LBPnNeighStart + LBPnNeighs-1);
        newLBPnNeighs = 2^(LBPnNeighs);
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
        newBSIFfSize = BSIFfSizeStart + (2 * (BSIFfSize - 1));
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
