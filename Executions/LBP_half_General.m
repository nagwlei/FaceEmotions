%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% -faces: Structure containing the emotion, etnicity, id etc.
% -newfaces: structure containing the half and the quarter iamges
% -images: Structure containing the images in .data(:,:,:,j) and the labels
%   of the images in .labels
% -nneighStart: Start of the number of neighbours for the table
% -nneighEnd: End of the number of neighbours for the table
% -rStart: Start of radius for the table
% -rEnd: End of radius for the table (and each of the radius will be 
%   calculated as a ^2)
% -CVO: The partitions for test and train

% Output:
% -myMatrixLBPhalf: Table with the LBP ith the given number of neighbours and
%   radius (the rows are the number of neighbours and the columns are 
%   the radius).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function myMatrixLBPhalf = LBP_half_General(faces, newfaces, images, nneighStart, nneighEnd, rStart, rEnd, CVO)
    nneighbours = nneighStart;
    radius = rStart;

    myMatrixLBPhalf = zeros((nneighEnd - nneighStart)+1, (rEnd - rStart)+1);

     for zx = 1:((rEnd - rStart)+1)
         for zy = 1:((nneighEnd - nneighStart)+1)
            % Extract LBP features
            for i=1:length(faces)
                %i;
                % Obtain selected image
                img = images.data(:,:,:,i);
                % Extract LBP features
                lbpimg = extractLBPFeatures(rgb2gray(uint8(img)),'Upright',false, ...
                    'CellSize', [16 16], 'NumNeighbors',nneighbours,'Radius',radius);
                half = newfaces{i}.half;
                lbphalfimg = extractLBPFeatures(rgb2gray(uint8(half)),'Upright',false, ...
                    'CellSize', [16 16], 'NumNeighbors',nneighbours,'Radius',radius);
                finallbp = horzcat(lbpimg, lbphalfimg);
                faces{i}.LBP = finallbp;
            end;
            
            disp(strcat('LBP          nNeighbours: ', int2str(nneighbours), ...
                ' radius: ', int2str(radius)));
            
            errLBP = zeros(CVO.NumTestSets, 1);

            for i = 1:CVO.NumTestSets
                trIdx = CVO.training(i);
                teIdx = CVO.test(i);

                TrLBP = zeros(sum(trIdx), length(faces{1}.LBP));
                TeLBP = zeros(sum(teIdx), length(faces{1}.LBP));
                
                tr = 0;
                te = 0;

                for j = 1:length(faces)
                    if (teIdx(j)>0)
                        te = te + 1;
                        TeLBP(te,:) = faces{j}.LBP;
                    else
                        tr = tr + 1;
                        TrLBP(tr,:) = faces{j}.LBP;
                    end;
                end;

                t = templateSVM( 'Standardize', 1 );
                Mdl = fitcecoc(TrLBP, images.labels(trIdx), 'Learners', t);
                ytestLBP = predict(Mdl, TeLBP);

                errLBP(i) = sum(ytestLBP~=images.labels(teIdx)');
            end;

            myMatrixLBPhalf(zy, zx) = sum(errLBP)/sum(CVO.TestSize);

            disp(strcat('MAE of       LBP:    ', sprintf('%f', myMatrixLBPhalf(zy, zx))))

            % Newline
            disp(' ')
            
            % Go to the next element of the table
            nneighbours = nneighbours + 2^(zy);
        end;
        nneighbours = nneighStart;
        radius = radius +1;
    end;
end
