%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% The source code for the bsif implementation has been downloaded from
% http://www.ee.oulu.fi/~jkannala/bsif/bsif.html


% Inputs:   
% -faces: Structure containing the emotion, etnicity, id etc.
% -newfaces: structure containing the half and the quarter images
% -images: Structure containing the images in .data(:,:,:,j) and the labels
%   of the images in .labels
% -fSizeStart: Start of the sizes size
% -fSizeEnd: fSizeStart + number of elements to work with
% -bitStart: Start of bit sizes
% -bitEnd: End of bit sizes
% -CVO: The partitions for test and train
% Output:
% -myMatrixBSIF: Table with the BSIF with the given number of bits and
%   filter size (the rows are the bits and the columns are the size of the
%   filters).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function myMatrixBSIF = BSIF_quart_General(faces, newfaces, images, ...
    fSizeStart, fSizeEnd, bitStart, bitEnd, CVO)
    fSize = fSizeStart;
    bits = bitStart;

    myMatrixBSIF = zeros((fSizeEnd - fSizeStart)+1, (bitEnd - bitStart)+1);

     for zx = 1:((bitEnd - bitStart)+1)
         for zy = 1:((fSizeEnd - fSizeStart)+1)
            % Extract BSIF features for each of the images
            for i=1:length(faces)
                % Obtain selected image
                if (length(size(images.data))>3)
                    aux = images.data(:,:,:,i);
                    img = rgb2gray(uint8(aux));
                    half = rgb2gray(uint8(newfaces{i}.half));
                    quart = rgb2gray(uint8(newfaces{i}.quarter));
                else
                    aux = images.data(:,:,i);
                    img = uint8(aux);
                    half = uint8(newfaces{i}.half);
                    quart = uint8(newfaces{i}.quarter);
                end
                
                % Extract BSIF features
                f = filesep;
                ICAtextureFiltersdir = strcat('bsif_code_and_data', f, ...
                    'texturefilters', f, 'ICAtextureFilters_', ...
                    num2str(fSize), 'x', num2str(fSize), '_', ...
                    num2str(bits), 'bit');
                
                % normalized BSIF code word histogram
                load(ICAtextureFiltersdir);
                
                bsifimg = bsif(double(img), ICAtextureFilters,'nh');
                
                bsifhalfimg = bsif(double(half), ICAtextureFilters,'nh');
                 
                bsifquartimg = bsif(double(quart), ICAtextureFilters,'nh');
                faces{i}.BSIF = horzcat(bsifimg, bsifhalfimg, bsifquartimg);
            end;

            disp(strcat('FILTERS      fSize:     ', num2str(fSize), 'x', ...
                num2str(fSize), ' bits: ', num2str(bits)));
            
            errBSIF = zeros(CVO.NumTestSets, 1);

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

                TrBSIF = zeros(sum(trIdx), length(faces{1}.BSIF));
                TeBSIF = zeros(sum(teIdx), length(faces{1}.BSIF));
                
                tr = 0;
                te = 0;
                                
                for j = 1:length(faces)
                    if (teIdx(j)>0)
                        te = te + 1;
                        TeBSIF(te,:) = faces{j}.BSIF;
                    else
                        tr = tr + 1;
                        TrBSIF(tr,:) = faces{j}.BSIF;
                    end;
                end;
                
                t = templateSVM( 'Standardize', 1 );
                Mdl = fitcecoc(TrBSIF, images.labels(trIdx), 'Learners', t);
                ytestBSIF = predict(Mdl, TeBSIF);

                errBSIF(i) = sum(ytestBSIF~=images.labels(teIdx)');
            end;

            % Introduce MAE in the matrix
            myMatrixBSIF(zy, zx) = sum(errBSIF)/sum(CVO.TestSize);

            disp(strcat('MAE of       BSIF:    ', sprintf('%f', myMatrixBSIF(zy, zx))))
            % Newline
            disp(' ')
            
            % Go to the next element of the table
            fSize = fSize + 2;
        end;
        fSize = fSizeStart;
        bits = bits +1;
    end;
end