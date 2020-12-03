function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


hypothsis = sigmoid(X * theta);

temp_theta = theta;
% 将theta(0)置成0,从而不会影响正则化带lambda的计算结果,因为正则化从theta(1)开始计算
temp_theta(1) = 0;
J = 1/m * sum(-y.*log(hypothsis) - (1 - y).*log(1 - hypothsis)) + lambda/(2*m) * sum(temp_theta.^2);

grad = 1/m * sum((hypothsis - y).*X) + lambda/m*temp_theta';


% for i = 1 : m,
% 	grad = grad + (hypothsis(i) - y(i)) * X(i,:)';
% end

% ta = [0; theta(2:end)];

% grad = 1/m*grad + lambda/m*ta;

% =============================================================

end
