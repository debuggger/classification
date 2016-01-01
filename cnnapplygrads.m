function net = cnnapplygrads(net, opts)
    for iter = 2 : numel(net.layers)
        if strcmp(net.layers{iter}.type, 'c')
            for j = 1 : numel(net.layers{iter}.a)
                for ii = 1 : numel(net.layers{iter - 1}.a)
                    net.layers{iter}.k{ii}{j} = net.layers{iter}.k{ii}{j} - opts.alpha * net.layers{iter}.dk{ii}{j};
                end
                net.layers{iter}.b{j} = net.layers{iter}.b{j} - opts.alpha * net.layers{iter}.db{j};
            end
        end
    end

    net.ffW = net.ffW - opts.alpha * net.dffW;
    net.ffb = net.ffb - opts.alpha * net.dffb;
end
