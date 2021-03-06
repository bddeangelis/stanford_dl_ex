function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
  
  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%

  % Append zeros to the theta values correponding to the zero'd class
  theta = [ theta, zeros(size(theta,1),1) ];

  % Calculate the hypothesized probabilities for each example
  h = exp(theta'*X);
  
  % Normalize all hypothesized probabilities for each example by dividing by the sum of probabilities for each example
  normH = bsxfun(@rdivide, h, sum(h,1));
  
  % Get the groundTruth associated with the labeled data
  groundTruth = full(sparse(y, 1:m, 1));
% % %   % Alternative
% % %   I = sub2ind(size(normH), y, 1:size(normH,2));
% % %   groundTruth = zeros(size(normH));
% % %   groundTruth(I) = 1;

  % Calculate the cost function
  f = -sum(sum(groundTruth .* log(normH)));
  
  % Calculate the gradient
  g = -X*(groundTruth-normH)';

  % Delete off the last column from g
  g = g(:,1:end-1);
  
  % Reshape the gradient into a vector  
  g=g(:); % make gradient a vector for minFunc
  
end

