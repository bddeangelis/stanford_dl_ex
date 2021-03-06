function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei); % Theta values are an aggregate of all the W and b elements
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1); % cell to store activations at each level
gradStack = cell(numHidden+1, 1); % cell to store the gradients of each level

% Define the number of examples in the current batch
m = size(data,2);

%% forward prop
%%% YOUR CODE HERE %%%
% Forward propagation's purpose is to give a hypothesis value for each example for comparison to ground truth

% Loop through each of the layers
for ii = 1:size(stack,1)
% for ii = 1:size(stack,1)-1    
    
    if ii == 1
        
        % Expand the bias unit to be the appropriate number of examples
        bias = repmat(stack{ii}.b, 1, size(data,2));
        
        % Calculate the activations for each neuron
        if strcmp(ei.activation_fun, 'logistic')
            
            hAct{ii} = sigmoid(stack{ii}.W * data + bias);

        elseif strcmp(ei.activation_fun, 'tanh')
%             hAct{ii} = ...
        elseif strcmp(ei.activation_fun, 'rectLin')
%             hAct{ii} = ...
        end
        
    % The output layer is different because of the softmax regression    
    elseif ii == size(stack,1)
        
        
        h = exp(stack{ii}.W*hAct{ii-1}); 
        hAct{ii} = bsxfun(@rdivide, h, sum(h,1));
        
    else
        
        % Expand the bias unit to be the appropriate number of examples
        bias = repmat(stack{ii}.b, 1, size(hAct{ii-1},2));  
        
        % Calculate the activations for each neuron
        if strcmp(ei.activation_fun, 'logistic')
            hAct{ii} = sigmoid(stack{ii}.W * hAct{ii-1} + bias);
        elseif strcmp(ei.activation_fun, 'tanh')
%             hAct{ii} = ...
        elseif strcmp(ei.activation_fun, 'rectLin')
%             hAct{ii} = ...
        end

    end

end

%% return here if only predictions desired.

if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];
  
  % NOTE: need to return the predicted probabilities from the forward run
  pred_prob = hAct{end}; % Test if this is correct later
%   pred_prob = normH; % It shouldn't actually matter which of these is used
  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%

  % Get the groundTruth associated with the labeled data
  groundTruth = full(sparse(labels, 1:size(data,2), 1));
      
  % Calculate the cross-entropy cost function
  ceCost = -sum(sum(groundTruth .* log(hAct{end})));
  
  
  %% compute gradients using backpropagation
  %%% YOUR CODE HERE %%%
  
  % Loop through each of the layers starting with the last layer to calculate the deltaErr terms for each of the associated edges
  for kk = size(stack,1):-1:1
      
      % CALCULATE THE ERROR TERMS
      % If you are at the output layer do something different
      if kk == size(stack,1)
          
          % Formula will be different depending on the chosen non-linearity
          if strcmp(ei.activation_fun, 'logistic')
              
              % Make the assignment for the error term
              deltaErr{kk} = -(groundTruth-hAct{kk});
              
          end
          
      % Condition for all hidden layers
      else
          
          % Formula for delta terms are different depending on the non-linearity selected
          if strcmp(ei.activation_fun, 'logistic')
              
              % Make the assignment for the error term
              deltaErr{kk} = stack{kk+1}.W' * deltaErr{kk+1} .* hAct{kk} .*(1-hAct{kk});
          end
      end
      
      % CALCULATE THE GRADIENTS
      % Weights for the input layer --> first hidden layer use the input data as activations  
      if kk == 1
          
          % Compute the associated gradients for W
          gradStack{kk}.W = deltaErr{kk} * data';
          
      % Weights for all other layer-to-layer connections use the calculated activations
      else
          
          % Compute the associated gradients for W
          gradStack{kk}.W = deltaErr{kk} * hAct{kk-1}';
          
      end
      
      % Compute the associated gradients for b
      gradStack{kk}.b = sum(deltaErr{kk},2);

  end
%% compute weight penalty cost

% Initialize a variable for the layer weights
totalW = 0;

% Extract the number of layers with weights
for jj = 1:size(stack,1)
    
    % Get the sum of the layer's weights
    layerW = sum(sum((stack{jj}.W).^2)); 
    
    % Add this to the total
    totalW = totalW + layerW;
    
end

% Use the learning rate stored in ei.lambda
wCost = ei.lambda/2 * totalW;  % Essentially, this is the sum of the squared weights for all weights in the network

% Sum together the components of the cost
cost = ceCost + wCost;

%% reshape gradients into vector
[grad] = stack2params(gradStack);

end



