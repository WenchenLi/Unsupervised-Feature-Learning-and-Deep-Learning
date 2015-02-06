function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 


    
%Feedforward and cost function

input_size = size(data,2);
B1 = repmat(b1,1,input_size);
B2 = repmat(b2,1,input_size);

z2 = W1 * data + B1;
a2 = sigmoid(z2);

z3 = W2 * a2 + B2;
a3 =  sigmoid(z3);
weight_decay = sum(sum(W1.^2))+sum(sum(W2.^2));
avg_activation_row = mean(a2,2);

KL_div = 0;
for j= 1: hiddenSize
    KL_div =KL_div + sparsityParam*log(sparsityParam/avg_activation_row(j))+(1-sparsityParam)*log((1-sparsityParam)/(1-avg_activation_row(j)));
end

raw_cost = (.5* input_size^(-1)) * sum(sum((a3-data).^2));
cost = raw_cost + (.5*lambda) * weight_decay + beta * KL_div;



%bp
delta_output = (a3-data).*(a3.*(ones(size(a3)) - a3));
delta_hidden = (W2'*delta_output + beta* ((-1)*sparsityParam*ones(size(a2))...
    ./(repmat(avg_activation_row,1,input_size)) + (1-sparsityParam)*ones(size(a2))./...
    (ones(size(a2))- repmat(avg_activation_row,1,input_size)))).*(a2.*(ones(size(a2))-a2));

dw2J = delta_output*a2';
db2J = delta_output;
dw1J = delta_hidden*data';
db1J = delta_hidden;

W1grad = dw1J/input_size + lambda * W1;
b1grad = mean(db1J,2) ;
W2grad = dw2J/input_size+ lambda * W2;
b2grad = mean(db2J,2) ;

% Delta_W2 = zeros(size(W2));
% Delta_b2 = zeros(size(b2));
% 
% Delta_W1 = zeros(size(W1));
% Delta_b1 = zeros(size(b1));
% 
% Delta_W2 = dw2J;
% Delta_b2 = db2J;
% 
% alpha = 0.05;
% W2 = W2 - alpha*((Delta_W2/input_size) + lambda*W2);
% b2 = b2 - alpha*mean(Delta_b2')';
% 
% Delta_W1 = dw1J;
% Delta_b1 = db1J;
% W1 = W1 - alpha*((Delta_W1/input_size) + lambda*W1);
% b1 = b1 - alpha*mean(Delta_b1')';













%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

