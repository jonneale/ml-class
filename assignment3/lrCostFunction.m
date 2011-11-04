function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
m = length(y);
'in here'
size(theta)
size(X)
size(y)
size(lambda)
% h0 = (sigmoid(X*theta));
% first = log(h0);
% second = 1-log(h0);
% J = (1/m)*sum((-y.*first)-((1-y).*second)) + lambda/(2*m)*sum(theta(2:end).^2);
% grad = ((1/m) * ((h0-y)' * X)' + vertcat(0,lambda*theta(2:end)))';
ths = sum(theta(:,2:size(theta)(2)).^2);
J = (1/m)*sum((-y.*log(sigmoid(X*theta)))-((1-y).*log(1-sigmoid(X*theta)))) + lambda/(2*m)*sum(theta(2:end).^2);
errors = sigmoid(X*theta)-y;
errors_matrix = repmat(errors,1,length(theta));

grad = (1/m)*(sum(errors_matrix.*X) + vertcat(0,lambda*theta(2:end))');


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%










% =============================================================

end
