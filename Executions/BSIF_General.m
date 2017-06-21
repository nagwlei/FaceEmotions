%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% The source code for the bsif implementation has been downloaded from
% http://www.ee.oulu.fi/~jkannala/bsif/bsif.html


% Inputs:
% -faces: 
% -images: Structure containing the images in .data(:,:,:,j) and the labels
%   of the images in .labels
% -fSizeStart: Start of the sizes size
% -fSizeEnd: fSizeStart + number of elements to work with
% -bitStart: Start of bit sizes
% -bitEnd: End of bit sizes
% -nFolds: The number of folds to do the CrossValidation

% Output:
% -myMatrixBSIF: Table with the BSIF with the given number of bits and
%   filter size (the rows are the bits and the columns are the size of the
%   filters).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function myMatrixBSIF = BSIF_General(faces, images, fSizeStart, fSizeEnd, bitStart, bitEnd, nfolds)
    fSize = fSizeStart;
    bits = bitStart;

    myMatrixBSIF = zeros((fSizeEnd - fSizeStart)+1, (bitEnd - bitStart)+1);

     for zx = 1:((bitEnd - bitStart)+1)
         for zy = 1:((fSizeEnd - fSizeStart)+1)
            % Extract BSIF features
            for i=1:length(faces)
                disp(strcat('i: ', int2str(i)));
                % Obtain selected image
                img = images.data(:,:,:,i);
                % Extract BSIF features
                ICAtextureFiltersdir = strcat('bsif_code_and_data\texturefilters\ICAtextureFilters_', num2str(fSize), 'x', num2str(fSize), '_', num2str(bits), 'bit');
                % normalized BSIF code word histogram
                load(ICAtextureFiltersdir);
                %bsifhistnnorm = bsif(double(rgb2gray(uint8(img))), ICAtextureFilters,'h');
                %faces{i}.BSIF = bsifhistnnorm;
                bsifhistnorm = bsif(double(rgb2gray(uint8(img))), ICAtextureFilters,'nh');
                faces{i}.BSIF = bsifhistnorm;
                %bsifcodeim = bsif(double(rgb2gray(uint8(img))),ICAtextureFilters,'im');
                %bsifcodeim = reshape(bsifcodeim, [], 1);
                %faces{i}.BSIF = bsifcodeim';
            end;

            disp(ICAtextureFiltersdir);
            
            % Create the folds
            CVO = cvpartition(images.labels, 'k', nfolds);
            %CVO = cvpartition(images.labels, 'k', 10);
            errBSIF = zeros(CVO.NumTestSets, 1);

            for i = 1:CVO.NumTestSets
                disp(strcat('test i: ', int2str(i)));
                TrBSIF = [];
                TsBSIF = [];

                trIdx = CVO.training(i);
                teIdx = CVO.test(i);

                for j = 1:length(faces)
                    %j
                    if (teIdx(j)>0)
                        %disp('IF');
                        TsBSIF = vertcat(TsBSIF, faces{j}.BSIF);
                    else
                        %disp('ELSE');
                        TrBSIF = vertcat(TrBSIF, faces{j}.BSIF);
                    end;
                end;


                Mdl = fitcecoc(TrBSIF, images.labels(trIdx));
                ytestBSIF = predict(Mdl, TsBSIF);

                errBSIF(i) = sum(ytestBSIF~=images.labels(teIdx)');
            end;

            cvErrBSIF = sum(errBSIF)/sum(CVO.TestSize);


            myMatrixBSIF(zy, zx) = cvErrBSIF;


            disp(strcat('VUELTA                zx:', int2str(zx), '    zy:', int2str(zy)))
            %disp(strcat('VALORRRRRRR                :    ', sprintf('%f', cvErrHOG)))
            disp(strcat('VALOR             BSIF:    ', sprintf('%f', cvErrBSIF)))
            % Go to the next element of the table
            fSize = fSize + 2;
            disp(strcat('CellSize  after     :', int2str(fSize)));
        end;
        fSize = fSizeStart;
        bits = bits +1;
    end;
end