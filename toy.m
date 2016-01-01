function [] = toy()

    im = load('newimages.mat');
    l = load('newlabels.mat');
    
    images = im.images;
    labels = l.labels;
    
    %logistic_regression(images, labels, 0.000001);
    
    images = [1 1 0 1]';
    labels = [1];
    neural_net(images, labels, 1);
    
end

function [] = logistic_regression(x, t, eta)
    [K, ~] = size(t);
    [M, N] = size(x);
    w = zeros(K, M);
    
    for iter=1:70
        for i=1:N
            phi = x(:,i);
            tn  = t(:,i);
            an = w*phi; %10x1
            yn = softmax_activation(an);
            for k=1:K
                deltaE = (yn(k) - tn(k))*phi;
                w(k, :) = w(k, :) - eta*deltaE'; 
            end
        end
    end
    res = w*x;
    
    result = softmax_activation(res);
    [~, class] = max(result);
    count = 0;
    l=load('labels.mat');
    labels = l.labels;
    
    for i=1:N
        if class(i)== labels(i) 
            count=count+1;
        end
        
        if class(i)-labels(i)==10
            count=count+1;
            class(i) = 0;
        end
    end
    count
    save('weight.mat', 'w', 'result', 'class');
    
end


function [] = neural_net(images, t, eta)
    eta=0.9;
    l=load('labels.mat');
    labels = l.labels;
    
    x=images;
    
    %{
    for i=images
        temp1 = reshape(i(2:length(i)), 28, 28);
        temp2 = [sum(temp1) sum(temp1')];
        x = [x [1 temp2]'];
    end
    %}
    
    [K, ~] = size(t);
    [M, N] = size(x);
    J = 2; %Number of units in hidden layer
    
    w1 = ones(J, M);
    w2 = ones(K, J+1);
    
    w1 = [-0.4 0.2  0.4 -0.5;
           0.2 -0.3 0.1 0.2];
       
    w2 = [0.1 -0.3 -0.2];
    
    for iter=1:1
        for i=1:N
            phi = x(:,i);
            tn  = t(:,i);
            labels(i);
            
            bn = w1*phi; %Jx1
            z=[1; h(bn)];
            
            an = w2*z;
            yn = sigmoid(an);

            sigmak = compute_sigmak(yn, tn);
            sigmaj = compute_sigmaj(z, w2, sigmak);
            
            w2 = update_w2(w2, sigmak, z, eta);
            w1 = update_w1(w1, sigmaj(2:J+1), phi, eta);
        end
    end
    w1
    w2
    %w1
    %w2
    
    count=0;
    [res, class] = NN_classify(x, w1, w2);
    for i=1:length(class)
        if class(i) == labels(i) 
            count=count+1;
        end
    end
    
    count
    save('nn_weights.mat', 'w1', 'w2', 'class', 'res');
    

end

function [yn] = softmax_activation(an)
    
    [a,b] = size(an);
    yn = zeros(a,b);
    
    denom = sum(exp(an));
    
    exp(an(:,1))./denom(1);
    for i=1:b
        yn(:,i) = exp(an(:,i))./denom(i);
    end
end

function [res] = h(bn)
    res = sigmoid(bn);
    %res = relu(bn);
end

function [res] = h_prime(zn)
    res = diff_sigmoid(zn);
    %res = diff_relu(zn);
end

function [res] = relu(a)
    res = arrayfun(@(x) max(0, x), a);
end

function [res] = diff_relu(a)
    res = arrayfun(@(x) x>0, a);
end

function [res] = sigmoid(a)
    res = arrayfun(@(x) 1/(1+exp(-1*x)), a);
end

function [res] = diff_sigmoid(z)
    res = arrayfun(@(x) (1/(1+exp(-1*x)))*(1 - (1/(1+exp(-1*x)))), z);
end

function [res] = compute_sigmak(yn, tn)
    res = yn*(1-yn)*(yn - tn);
end

function [res] = compute_sigmaj(z, w2, sigmak)
    [J, ~] = size(z);
    res = zeros(J, 1);
    
    for j=1:J
        weighted_error = sum(w2(j)' * sigmak);
        res(j) = z(j)*(1-z(j))* weighted_error;
    end
end


function [updated_w1] = update_w1(w1, sigmaj, phi, eta)
    dw1 = sigmaj*phi';
    updated_w1 = w1 - eta*dw1;
end

function [updated_w2] = update_w2(w2, sigmak, z, eta)
    dw2 = sigmak*z';
    updated_w2 = w2 - eta*dw2;
end

function [res, class_result] = NN_classify(x, w1, w2)
    [~, N] = size(x);
    z = [ones(1, N); h(w1*x)]; %hidden layer
    res = softmax_activation(w2*z);
    [~, class_result] = max(res);
    
    for i=1:length(class_result)
        if class_result(i)==10
            class_result(i) =  0;
        end
    end
end
