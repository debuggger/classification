function net = cnntrain(net, x, y, opts, test_x, test_y)
    m = size(x, 3);
    numbatches = m / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL = [];
    for i = 1 : opts.numepochs
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        tic;
        kk = randperm(m);
        for iter = 1 : numbatches
            batch_x = x(:, :, kk((iter - 1) * opts.batchsize + 1 : iter * opts.batchsize));
            batch_y = y(:,    kk((iter - 1) * opts.batchsize + 1 : iter * opts.batchsize));

            net = cnnff(net, batch_x);
            net = cnnbp(net, batch_y);
            net = cnnapplygrads(net, opts);
            if isempty(net.rL)
                net.rL(1) = net.L;
            end
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
        end
        toc;
		%if(mod(i,2) == 0)
			[er, bad] = cnntest(net, test_x, test_y);
            disp('==============================');
			fprintf('Epoch=%d error=%f\n', i, er);
            disp('==============================');
		%end
        
    end
    
end
