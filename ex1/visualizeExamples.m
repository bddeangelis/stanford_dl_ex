function visualizeExamples( data, labels, pred )
% This function visualizes some examples and outputs the associated labels

% reshape the data
dataImage = reshape(data, [28 28 size(data,2)]);

% Generate 9 random indices between 1 and the size of the dataset 
ind = round(rand(9,1)*size(data,2));

for n = 1:9
    
    % Show the digit
    subplot(3,3,n);
    imshow(dataImage(:,:,ind(n)));
    title(strcat('Predicted: ',num2str(pred(ind(n))-1))); % NOTE need to change the indexing from 1-10 to 0-9

    
end


end

