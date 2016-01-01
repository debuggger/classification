function [yn] = activation(an)
    
    [a,b] = size(an);
    yn = zeros(a,b);
    for i=1:b
        denom = sum(exp(an(:,i)));
        for j=1:a
            yn(j,i) = exp(an(j,i))/denom;
        end
    end
end