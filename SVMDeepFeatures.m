function MAEDeep = SVMDeepFeatures( CVO, faces, images )

    for i = 1:CVO.NumTestSets
        trIdx = CVO.training(i);
        teIdx = CVO.test(i);

        % This is necessary to work with the CVO created for JAFFE
        if (iscell(trIdx))
           trIdx = trIdx{1}; 
        end
        if (iscell(teIdx))
            teIdx = teIdx{1};
        end

        TrDeep = zeros(sum(trIdx), length(faces{1}.deep_features));
        TeDeep = zeros(sum(teIdx), length(faces{1}.deep_features));

        tr = 0;
        te = 0;

        for j = 1:length(faces)
            if (teIdx(j)>0)
                te = te + 1;
                TeDeep(te,:) = faces{j}.deep_features;
            else
                tr = tr + 1;
                TrDeep(tr,:) = faces{j}.deep_features;
            end
        end

        t = templateSVM( 'Standardize', 1 );
        Mdl = fitcecoc(TrDeep, images.labels(trIdx), 'Learners', t);
        ytestDeep = predict(Mdl, TeDeep);

        errDeep(i) = sum(ytestDeep~=images.labels(teIdx)');
    end

    MAEDeep = sum(errDeep)/sum(CVO.TestSize);

    

end