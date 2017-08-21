function facesWithDeepFeatures = getMatConvNetFeatures( netPath, gpu, faces, images )

    % setup MatConvNet (needs to have MatConvNet in the path)
    run vl_setupnn

    % read network
    net = load( netPath ) ;

    net = vl_simplenn_tidy( net );
    % calculate in gpu
    if( gpu )
        net = vl_simplenn_move( net, 'gpu' );
    end

    tic;

    if (length(size(images.data))>3)
        colorImages = 1;
    else
        colorImages = 0;
    end
                    
    
    for kk = 1:length( faces )
        %disp( kk );
        % apply network to image

        % read image
        if( colorImages )
            im = images.data(:,:,:, kk);
        else
            im = images.data(:,:,kk);
            im = cat(3, im, im, im);
        end
        % note: 255 range
        im_ = single(im) ;
        % resize to 224x224 (expected input size in the network)
        im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
        % normalize
        im_ = bsxfun(@minus,im_,net.meta.normalization.averageImage) ;
        % apply net
        if( gpu )
            res = vl_simplenn(net, gpuArray( im_ ) );
            % extract 4K features
            feat4K = reshape( gather(res(end-2).x), 4096, 1);
        else
            res = vl_simplenn(net, im_ ) ;
            % extract 4K features
            feat4K = reshape( res(end-2).x, 4096, 1);
        end

        % apply L2 norm to features
        faces{ kk }.deep_features = feat4K / norm( feat4K, 2 );

    end

    toc;
    
    facesWithDeepFeatures = faces;

end