function [res1, res2] = test(a)
    sigmoid(a)
    
    diff_sigmoid(a)
end

function [res] = sigmoid(a)
    res = arrayfun(@(x) 1/(1+exp(-1*x)), a);
end

function [res] = diff_sigmoid(z)
    res = arrayfun(@(x) (1/(1+exp(-1*x)))*(1 - (1/(1+exp(-1*x)))), z);
end