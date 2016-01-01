function [er, bad] = cnntest(net, x, y)
    %  feedforward
    net = cnnff(net, x);
    [~, h] = max(net.o);
    [~, a] = max(y);
    bad = find(h ~= a);
    size(y, 2)
    h
    size(y)
    size(net.o)
    net.o(1:10)
    er = numel(bad) / size(y, 2);
end
