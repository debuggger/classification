function net = cnnff(net, x)
    n = numel(net.layers);
    net.layers{1}.a{1} = x;
    inputmaps = 1;
    
    net.layers{1}.a;

    for iter = 2 : n   %  for each layer
        if strcmp(net.layers{iter}.type, 'c')
            %  !!below can probably be handled by insane matrix operations
            for j = 1 : net.layers{iter}.outputmaps   %  for each output map
                %  create temp output map
                z = zeros(size(net.layers{iter - 1}.a{1}) - [net.layers{iter}.kernelsize - 1 net.layers{iter}.kernelsize - 1 0]);
                for i = 1 : inputmaps   %  for each input map
                    %  convolve with corresponding kernel and add to temp output map
                    net.layers{1}.a{i}
					z = z + convn(net.layers{iter - 1}.a{i}, net.layers{iter}.k{i}{j}, 'valid');
                end
                %  add bias, pass through nonlinearity
                net.layers{iter}.a{j} = sigm(z + net.layers{iter}.b{j});
            end
            %  set number of input maps to this layers number of outputmaps
            inputmaps = net.layers{iter}.outputmaps;
        elseif strcmp(net.layers{iter}.type, 's')
            %  downsample
            for j = 1 : inputmaps
                z = convn(net.layers{iter - 1}.a{j}, ones(net.layers{iter}.scale) / (net.layers{iter}.scale ^ 2), 'valid');   %  !! replace with variable
                net.layers{iter}.a{j} = z(1 : net.layers{iter}.scale : end, 1 : net.layers{iter}.scale : end, :);
            end
        end
    end

    %  concatenate all end layer feature maps into vector
    net.fv = [];
    for j = 1 : numel(net.layers{n}.a)
        sa = size(net.layers{n}.a{j});
        net.fv = [net.fv; reshape(net.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
    end
    %  feedforward into output perceptrons
    net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));

end
