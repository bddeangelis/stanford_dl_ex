function [f,g] = linear_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The target value for each example.  y(j) is the target for example j.
  %
  m=size(X,2);
  n = size(X,1);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the linear regression objective function and gradient 
  %        using vectorized code.  (It will be just a few lines of code!)
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %
%%% YOUR CODE HERE %%%

% Calculate an intermediate value used in both f and g
z = theta'*X-y;

% Compute the cost function
f = .5*sum(z.^2);

% Compute the gradient of the cost function
% g = sum(X.*repmat(z,n,1),2);
g = X*z';


end