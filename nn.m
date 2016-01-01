function [] = nn()

	data = load('train_data.mat');
    train_x = data.x(2:end, :);
    train_y = data.t;
	
	losses = []; train_errors = []; test_wrongs = [];  
	%Here we perform mini-batch stochastic gradient descent %If batchsize = 1, it would be stochastic gradient descent %If batchsize = N, it would be basic gradient descent  
	batchsize = 50;  
	%Num of batches  
	numbatches = size(train_x, 2) / batchsize;  
	%% Training part %Learning rate alpha 
	alpha = 0.01;  
	%Lambda is for regularization 
	lambda = 0.001;   
	%Num of iterations 
	numepochs = 20; 
    numOfHiddenLayer = 2;

	s{1} = size(train_x, 1); s{2} = 200; s{3} = 10; s{4} = 100; s{5} = 10;
	
	for i = 1 : numOfHiddenLayer     
		W{i} = zeros(s{i+1}, s{i});     
		b{i} = 0; 
	end 

	for j = 1 : numepochs     %randomly rearrange the training data for each epoch     %We keep the shuffled index in kk, so that the input and output could      %be matched together      
		kk = randperm(size(train_x, 2));  
		for l = 1 : numbatches  
			%Set the activation of the first layer to be the training data         %while the target is training labels  
			a{1} = train_x(:, kk((l-1)*batchsize+1 : l*batchsize ));
			y = train_y(:, kk((l-1)*batchsize+1 : l*batchsize ));  
			%Forward propagation, layer by layer        %Here we use sigmoid function as an example  
			for i = 2 : numOfHiddenLayer + 1
                a{i} = sigm( bsxfun(@plus, W{i-1}*a{i-1}, b{i-1}) );
			end
			
            %Calculate the error and back-propagate error layer by layers
            d{numOfHiddenLayer + 1} =  -(y - a{numOfHiddenLayer + 1}) .* a{numOfHiddenLayer + 1} .* (1-a{numOfHiddenLayer + 1});  
			for i = numOfHiddenLayer : -1 : 2             
				d{i} = W{i}' * d{i+1} .* a{i} .* (1-a{i});         
			end                  %Calculate the gradients we need to update the parameters         %L2 regularization is used for W  
			for i = 1 : numOfHiddenLayer             
				dW{i} = d{i+1} * a{i}';
				db{i} = sum(d{i+1}, 2);             
				W{i} = W{i} - alpha * (dW{i} + lambda * W{i});             
				b{i} = b{i} - alpha * db{i};         
			end     
		end 
    end

    [~, cr] = NN_classify(train_x, W, b, numOfHiddenLayer);
    [acc, ~, ~] = accuracy(data.labels, cr);
    acc
    
end

function [acc, correct, wrong] = accuracy(labels, class_result)
    count = 0;
    total_count=length(labels);
    
    for i=1:length(labels)
        if labels(i)==class_result(i)
            count = count+1;
        end
    end
    correct = count;
    wrong = total_count-correct;
    acc = correct*100/total_count;
end


function [res, class_result] = NN_classify(x, W, b, numOfHiddenLayer)

    a{1} = x;
    
    for i = 2 : numOfHiddenLayer + 1
        a{i} = sigm( bsxfun(@plus, W{i-1}*a{i-1}, b{i-1}) );
    end
	res = 0;
    [~, class_result] = max(a{numOfHiddenLayer + 1});
    class_result = mod(class_result, 10);
    save('temp_res', 'class_result');
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
    res = sigm(bn);
    %res = relu(bn);
end