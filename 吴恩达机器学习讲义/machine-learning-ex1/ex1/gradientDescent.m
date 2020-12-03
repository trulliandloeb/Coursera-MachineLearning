function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % hypothesis: h_theta(x)
    predictions = X*theta;
    % gradient descent
    % (h_theta(x) - y) * x(j)
    sqrErrors = (predictions-y).*X;
    % alpha * 1/m * sigama_i_from_1_to_m
    dJ = 1/(m) * sum(sqrErrors);
    theta = theta - alpha*dJ';

    %theta = theta - alpha*X'*(X*theta - y)/m;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
