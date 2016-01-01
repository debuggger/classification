function net = cnnbp(net, y)
    n = numel(net.layers);

    %   error
    net.e = net.o - y;
    %  loss function
    net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);

    %%  backprop deltas
    net.od = net.e .* (net.o .* (1 - net.o));   %  output delta
    net.fvd = (net.ffW' * net.od);              %  feature vector delta
    if strcmp(net.layers{n}.type, 'c')         %  only conv layers has sigm function
        net.fvd = net.fvd .* (net.fv .* (1 - net.fv));
    end

    %  reshape feature vector deltas into output map style
    sa = size(net.layers{n}.a{1});
    fvnum = sa(1) * sa(2);
    for j = 1 : numel(net.layers{n}.a)
        net.layers{n}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
    end

    for iter = (n - 1) : -1 : 1
        if strcmp(net.layers{iter}.type, 'c')
            for j = 1 : numel(net.layers{iter}.a)
                net.layers{iter}.d{j} = net.layers{iter}.a{j} .* (1 - net.layers{iter}.a{j}) .* (expand(net.layers{iter + 1}.d{j}, [net.layers{iter + 1}.scale net.layers{iter + 1}.scale 1]) / net.layers{iter + 1}.scale ^ 2);
            end
        elseif strcmp(net.layers{iter}.type, 's')
            for i = 1 : numel(net.layers{iter}.a)
                z = zeros(size(net.layers{iter}.a{1}));
                for j = 1 : numel(net.layers{iter + 1}.a)
                     z = z + convn(net.layers{iter + 1}.d{j}, rot180(net.layers{iter + 1}.k{i}{j}), 'full');
                end
                net.layers{iter}.d{i} = z;
            end
        end
    end

    %%  calc gradients
    for iter = 2 : n
        if strcmp(net.layers{iter}.type, 'c')
            for j = 1 : numel(net.layers{iter}.a)
                for i = 1 : numel(net.layers{iter - 1}.a)
                    net.layers{iter}.dk{i}{j} = convn(flipall(net.layers{iter - 1}.a{i}), net.layers{iter}.d{j}, 'valid') / size(net.layers{iter}.d{j}, 3);
                end
                net.layers{iter}.db{j} = sum(net.layers{iter}.d{j}(:)) / size(net.layers{iter}.d{j}, 3);
            end
        end
    end
    net.dffW = net.od * (net.fv)' / size(net.od, 2);
    net.dffb = mean(net.od, 2);

    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
end
