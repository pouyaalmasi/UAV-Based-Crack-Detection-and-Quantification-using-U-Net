clc; clear; close all;

%% **Step 1: Define Paths**
imageDir = 'images';  
maskDir = 'masks';    
outputDir = 'dataset/resized'; 
mkdir(outputDir);
mkdir(fullfile(outputDir, 'images'));
mkdir(fullfile(outputDir, 'masks'));

%% **Step 2: Create Image and Mask Datastores**
imdsOriginal = imageDatastore(imageDir, 'FileExtensions', '.jpg'); 

classNames = ["Crack", "Background"];
pixelLabelIDs = [1, 0];

pxdsOriginal = pixelLabelDatastore(maskDir, classNames, pixelLabelIDs);

%  Fix 7: Convert Masks to Binary (0 and 1)
numMasks = numel(pxdsOriginal.Files);
for i = 1:numMasks
    mask = imread(pxdsOriginal.Files{i});
    
    % Convert grayscale or incorrect masks to binary
    mask = im2bw(mask, 0.5);  % Convert using threshold 0.5
    mask(mask > 0) = 1;  % Ensure all nonzero pixels are 1
    mask(mask == 0) = 0; % Ensure background remains 0

    % Overwrite the corrected mask file
    imwrite(mask, pxdsOriginal.Files{i});
end

disp('* All masks converted to binary format!');

%% **Step 3: Extract File Paths & Split Data**
imageFiles = imdsOriginal.Files;
maskFiles = pxdsOriginal.Files;
numFiles = numel(imageFiles);

trainRatio = 0.88888;
numTrain = round(trainRatio * numFiles);
randIndices = randperm(numFiles);
trainIndices = randIndices(1:numTrain);
valIndices = randIndices(numTrain+1:end);

imdsTrainFiles = imageFiles(trainIndices);
imdsValFiles = imageFiles(valIndices);
pxdsTrainFiles = maskFiles(trainIndices);
pxdsValFiles = maskFiles(valIndices);

disp('* Data splitting complete!');

%% **Step 4: Resize Images & Masks**
imageSize = [224 224];

trainImagePaths = cell(numTrain, 1);
trainMaskPaths = cell(numTrain, 1);

for i = 1:numTrain
    img = imread(imdsTrainFiles{i});
    mask = imread(pxdsTrainFiles{i});
    
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    img = imresize(img, imageSize);
    mask = imresize(mask, imageSize, 'nearest');
    
    imgPath = fullfile(outputDir, 'images', sprintf('train_%d.jpg', i));
    maskPath = fullfile(outputDir, 'masks', sprintf('train_%d.png', i));
    imwrite(img, imgPath);
    imwrite(mask, maskPath);
    
    trainImagePaths{i} = imgPath;
    trainMaskPaths{i} = maskPath;
end

disp('* Train images and masks resized and saved!');
figure;
for i = 1:3
    subplot(1,3,i);
    img = imread(trainMaskPaths{i});
    imshow(img);
    title(sprintf('Training Mask %d', i));
end

numVal = numFiles - numTrain;
valImagePaths = cell(numVal, 1);
valMaskPaths = cell(numVal, 1);

for i = 1:numVal
    img = imread(imdsValFiles{i});
    mask = imread(pxdsValFiles{i});
    
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    img = imresize(img, imageSize);
    mask = imresize(mask, imageSize, 'nearest');
    
    imgPath = fullfile(outputDir, 'images', sprintf('val_%d.jpg', i));
    maskPath = fullfile(outputDir, 'masks', sprintf('val_%d.png', i));
    imwrite(img, imgPath);
    imwrite(mask, maskPath);
    
    valImagePaths{i} = imgPath;
    valMaskPaths{i} = maskPath;
end

disp('* Validation images and masks resized and saved!');

%% **Step 5: Create Datastores**
imdsTrain = imageDatastore(trainImagePaths);
pxdsTrain = pixelLabelDatastore(trainMaskPaths, classNames, pixelLabelIDs);
imdsVal = imageDatastore(valImagePaths);
pxdsVal = pixelLabelDatastore(valMaskPaths, classNames, pixelLabelIDs);

disp('* Resized images and masks stored successfully!');

%% **Step 6: Data Augmentation**
augmenter = imageDataAugmenter( ...
    'RandRotation', [-15, 15], ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandScale', [0.9, 1.1], ...
    'RandXTranslation', [-10, 10], ...
    'RandYTranslation', [-10, 10] ...
);

trainData = pixelLabelImageDatastore(imdsTrain, pxdsTrain, 'DataAugmentation', augmenter);
valData = pixelLabelImageDatastore(imdsVal, pxdsVal);

disp('* Data successfully resized and combined!');

%% **Step 7: Define U-Net Architecture**
numClasses = 2;  
lgraph = unetLayers(imageSize, numClasses);

lgraph = replaceLayer(lgraph, 'ImageInputLayer', ...
    imageInputLayer([224 224 1], 'Name', 'input', 'Normalization', 'none'));

% Display the layer names  (debug)
% analyzeNetwork(lgraph)

dropout1 = dropoutLayer(0.2, 'Name', 'dropout1');
dropout2 = dropoutLayer(0.3, 'Name', 'dropout2');
dropout3 = dropoutLayer(0.3, 'Name', 'dropout3');

lgraph = replaceLayer(lgraph, 'Encoder-Stage-1-ReLU-2', ...
    [reluLayer('Name', 'Encoder-Stage-1-ReLU-2'); dropout1]);
lgraph = replaceLayer(lgraph, 'Encoder-Stage-2-ReLU-2', ...
    [reluLayer('Name', 'Encoder-Stage-2-ReLU-2'); dropout2]);
lgraph = replaceLayer(lgraph, 'Encoder-Stage-3-ReLU-2', ...
    [reluLayer('Name', 'Encoder-Stage-3-ReLU-2'); dropout3]);

disp('* Dropout layers added successfully!');

%% **Step 8: Class Weighting & Loss Function**
classWeights = [100, 1]; 
pxLayer = pixelClassificationLayer('ClassNames', classNames, 'ClassWeights', classWeights);
lgraph = replaceLayer(lgraph, 'Segmentation-Layer', pxLayer);

disp('* U-Net architecture created successfully!');

%% **Step 9: Training Options**
options = trainingOptions('adam', ...
    'L2Regularization', 1e-3, ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 8, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', valData, ...
    'ValidationFrequency', 10, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.8, ...
    'LearnRateDropPeriod', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

disp('* Training options set successfully!');

%% **Step 10: Train U-Net**
net = trainNetwork(trainData, lgraph, options);

%% **Step 11: Save Model**
save('trained_unet.mat', 'net');

disp('* U-Net training complete! Model saved.');
