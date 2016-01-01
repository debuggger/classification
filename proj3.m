function [] = proj3()

    train_x = loadMNISTImages('train-images-idx3-ubyte');
    train_labels = loadMNISTLabels('train-labels-idx1-ubyte');
    train_t = zeros(10, size(train_labels,1));
    
    test_x = loadMNISTImages('t10k-images.idx3-ubyte');
    test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
    test_t = zeros(10, size(train_labels,1));
    
    for i=1:size(train_labels,1)
        train_t(train_labels(i)+1,i) = 1;
    end
    
    for i=1:size(test_labels,1)
        test_t(test_labels(i)+1,i) = 1;
    end
    
    train_x = [ones(1,size(train_x, 2)); train_x];
    test_x = [ones(1,size(test_x, 2)); test_x];
    
    train = struct('x', train_x, 't', train_t, 'labels', train_labels);
    test = struct('x', test_x, 't', test_t, 'labels', test_labels);
    
    [w] = logistic_regression(train, test);
    
    %[w1, w2] = neural_net(train, test);
    
    Wlr = w(:, 2:end)';
    blr = w(:,1)';
    %{
    Wnn1 = w1(:, 2:end)';
    bnn1 = w1(:, 1)';
    Wnn2 = w2(:, 2:end)';
    bnn2 = w2(:, 1)';
    h = 'sigmoid';
    %}
    save('proj3.mat', 'Wlr', 'blr');
    %save('proj3.mat', 'Wnn1', 'bnn1', 'Wnn2', 'bnn2', 'h');
    %save('proj3.mat', 'Wlr', 'blr', 'Wnn1', 'bnn1', 'Wnn2', 'bnn2', 'h');
    
end

function [wmax] = logistic_regression(train, test)
    [K, ~] = size(train.t);
    [M, N] = size(train.x);
    w = rand(K, M) - rand(K, M);
    agg_result = [];
    max_acc = 0;
    
    for eta=[0.001 0.0025 0.005 0.0075]
    %for eta = 0.05
        tic;
        for iter=1:50
            for i=1:N
                phi = train.x(:,i);
                tn  = train.t(:,i);
                an = w*phi; %10x1
                yn = softmax_activation(an);

                deltaE =  (yn - tn) * phi';
                w = w - eta*deltaE;
            end
            if(mod(iter, 5) == 0)
                [~, class_train] = LR_classify(train.x, w);
                [acc_train, ~, ~] = accuracy(class_train, train.labels');
                fprintf('LR: Train eta=%f Iter:%d acc=%f \n', eta, iter, acc_train);

                [~, class_test] = LR_classify(test.x, w);
                [acc_test, ~, ~] = accuracy(class_test, test.labels');
                fprintf('LR: Test eta=%f Iter:%d acc=%f\n', eta, iter, acc_test);

                temp.eta = eta;
                temp.train_per = acc_train;
                temp.test_per = acc_test;
                temp.iterations = iter;
                temp.eta = eta;
                temp.w = w;
                agg_result = [agg_result; temp];
                save('results_lr_rand', 'agg_result');
                if (acc_test > max_acc)
                    max_acc = acc_test;
                    max_eta = eta;
                    max_iter = iter;
                    wmax = w;
                end
            end
        end
        toc;
    end
    fprintf('Max acc: %f Eta: %f Iter: %d\n', max_acc, max_eta, max_iter);
end


function [w1, w2] = neural_net(train, test)
    [K, ~] = size(train.t);
    [M, N] = size(train.x);
    agg_result = [];
    
    
    for J=[10 50 100 150 200 270 340 360 380 400]
            w1 = rand(J, M) - 0.4;
            w2 = rand(K, J+1) - 0.4;
        for eta=[0.001 0.005 0.01 0.05 0.1 0.5]
            tic;
            fprintf('J=%d eta=%f\n', J, eta);
            for iter=1:2
                for i=1:N
                %for i=randperm(N)
                    phi = train.x(:,i);
                    tn  = train.t(:,i);

                    bn = w1*phi; %Jx1
                    z = h(bn);
                    z=[1 z']';

                    an = w2*z;
                    %yn = softmax_activation(an);
                    yn = sigm(an);

                    deltak = compute_deltak(yn, tn);
                    deltaj = compute_deltaj(z, w2, deltak);

                    w2 = update_w2(w2, deltak, z, eta);
                    w1 = update_w1(w1, deltaj(2:J+1), phi, eta);
                end
            end
            disp('Starting classification')
            
            [~, class_train] = NN_classify(train.x, w1, w2);
            [acc_train, ~, ~] = accuracy(class_train, train.labels');
            fprintf('NN: Train eta=%f J=%d acc=%f \n', eta, J, acc_train);
            
            [~, class_test] = NN_classify(test.x, w1, w2);
            [acc_test, ~, ~] = accuracy(class_test, test.labels');
            fprintf('NN: Test eta=%f J=%d acc=%f\n', eta, J, acc_test);
            toc;
            
            temp.eta = eta;
            temp.hidden_node_count = J;
            temp.train_per = acc_train;
            temp.test_per = acc_test;
            agg_result = [agg_result; temp];
            save('results_nn', 'agg_result');
            
        end
    end
end

function [yn] = softmax_activation(an)
    
    [a,b] = size(an);
    yn = zeros(a,b);
    denom = sum(exp(an));
    
    for i=1:b
        yn(:,i) = exp(an(:,i))./denom(i);
    end
end

function [res] = h(bn)
    %res = sigm(bn);
    res = tanh(bn);
    %res = relu(bn);
end

function [res] = h_prime(zn)
    %res = diff_sigmoid(zn);
    res = diff_tanh(zn);
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
    res = sigm(z).*(1 - sigm(z));
end

function [res] = diff_tanh(z)
    res = 1 - tanh(z).^2;
end

function [res] = compute_deltak(yn, tn)
    res = (yn - tn);
end

function [res] = compute_deltaj(z, w2, deltak)
    [J, ~] = size(z);
    res = zeros(J, 1);
    gradZ = z .* (1 - z);
    
    weighted_error =  w2'*deltak;
    res = gradZ .* weighted_error;
    %{
    for j=1:J
        weighted_error = sum(w2(:,j)'*deltak);
        res(j) = gradZ(j)*weighted_error;
    end
    %}
    
end


function [updated_w1] = update_w1(w1, deltaj, phi, eta)
    dw1 = deltaj*phi';
    updated_w1 = w1 - eta*dw1;
end

function [updated_w2] = update_w2(w2, deltak, z, eta)
    dw2 = deltak*z';
    updated_w2 = w2 - eta*dw2;
end

function [res, class_result] = NN_classify(x, w1, w2)
    [~, N] = size(x);
    z = [ones(1, N); h(w1*x)]; %hidden layer
    res = softmax_activation(w2*z);
    [~, class_result] = max(res);
    class_result = class_result-1;
end

function [res, class_result] = LR_classify(x, w)
    res = w*x;
    result = softmax_activation(res);
    [~, class_result] = max(result);
    class_result = class_result-1;
end

function [acc, correct, wrong] = accuracy(labels, class_result)
    total_count=length(labels);
    
    correct = sum(labels == class_result);
    wrong = total_count-correct;
    acc = correct*100/total_count;
end