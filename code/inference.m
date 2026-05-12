%% inference.m
% Crack Segmentation and Quantification using Trained U-Net
%
% Author: Pouya Almasi
%
% Description:
% This script loads a trained U-Net model, performs crack segmentation on an
% input image, generates a binary crack mask and overlay, and computes crack
% quantification metrics including crack density, crack width, crack length,
% connected cracks, and severity classification.

clc;
clear;
close all;

%% User Inputs

modelPath = "model/trained_unet.mat";      % Path to trained U-Net model
imagePath = "sample_images/test.jpg";      % Input image
outputDir = "results";                     % Output folder

%% Create Output Directory

if ~exist(outputDir, "dir")
    mkdir(outputDir);
end

%% Load Trained Network

data = load(modelPath);

if isfield(data, "net")
    net = data.net;
else
    error("The MAT file must contain a trained network variable named 'net'.");
end

%% Read Input Image

imgOriginal = imread(imagePath);

% Convert RGB to grayscale if necessary
if size(imgOriginal, 3) == 3
    imgGray = rgb2gray(imgOriginal);
else
    imgGray = imgOriginal;
end

%% Store Original Size

originalSize = size(imgGray);

%% Resize for Network Input

inputSize = [224 224];

imgResized = imresize(imgGray, inputSize);

%% Perform Semantic Segmentation

predictedLabels = semanticseg(imgResized, net);

% Crack class name must match training labels
crackMaskSmall = predictedLabels == "Crack";

%% Resize Mask Back to Original Resolution

crackMask = imresize(crackMaskSmall, originalSize, "nearest");

%% Post-Processing

% Remove small noise regions
crackMask = bwareaopen(crackMask, 20);

% Morphological closing for crack continuity
crackMask = imclose(crackMask, strel("disk", 1));

%% Crack Quantification

% Crack Density
totalPixels = numel(crackMask);
crackPixels = nnz(crackMask);

crackDensity = (crackPixels / totalPixels) * 100;

%% Skeletonization

skeletonMask = bwmorph(crackMask, "skel", Inf);

%% Total Crack Length

totalCrackLength_px = nnz(skeletonMask);

%% Connected Crack Analysis

cc = bwconncomp(skeletonMask);

numCracks = cc.NumObjects;

if numCracks > 0

    crackLengths = cellfun(@numel, cc.PixelIdxList);

    longestCrack_px = max(crackLengths);

else

    longestCrack_px = 0;

end

%% Crack Width Estimation

distMap = bwdist(~crackMask);

widthValues = 2 * distMap(skeletonMask);

if isempty(widthValues)

    meanCrackWidth_px = 0;
    maxCrackWidth_px  = 0;

else

    meanCrackWidth_px = mean(widthValues);
    maxCrackWidth_px  = max(widthValues);

end

%% Severity Classification

if crackDensity < 2

    severity = "Low";

elseif crackDensity < 5

    severity = "Moderate";

else

    severity = "Severe";

end

%% Create Crack Overlay

overlay = labeloverlay( ...
    imgGray, ...
    crackMask, ...
    "Transparency", 0.55, ...
    "Colormap", [1 0 0]);

%% Display Results

figure("Name", "Crack Segmentation Results");

subplot(1,3,1);
imshow(imgOriginal);
title("Original Image");

subplot(1,3,2);
imshow(crackMask);
title("Binary Crack Mask");

subplot(1,3,3);
imshow(overlay);
title("Crack Overlay");

%% Save Outputs

[~, imageName, ~] = fileparts(imagePath);

% Save Binary Mask
imwrite( ...
    crackMask, ...
    fullfile(outputDir, imageName + "_binary_mask.png"));

% Save Overlay
imwrite( ...
    overlay, ...
    fullfile(outputDir, imageName + "_overlay.png"));

%% Save Quantification Report

summaryFile = fullfile( ...
    outputDir, ...
    imageName + "_crack_summary.txt");

fid = fopen(summaryFile, "w");

fprintf(fid, "Crack Segmentation and Quantification Results\n");
fprintf(fid, "--------------------------------------------\n");

fprintf(fid, "Image: %s\n", imagePath);

fprintf(fid, ...
    "Image size: %d x %d pixels\n", ...
    originalSize(1), ...
    originalSize(2));

fprintf(fid, ...
    "Crack density: %.4f %%\n", ...
    crackDensity);

fprintf(fid, ...
    "Mean crack width: %.4f pixels\n", ...
    meanCrackWidth_px);

fprintf(fid, ...
    "Maximum crack width: %.4f pixels\n", ...
    maxCrackWidth_px);

fprintf(fid, ...
    "Total crack length: %.2f pixels\n", ...
    totalCrackLength_px);

fprintf(fid, ...
    "Longest continuous crack: %.2f pixels\n", ...
    longestCrack_px);

fprintf(fid, ...
    "Number of connected cracks: %d\n", ...
    numCracks);

fprintf(fid, ...
    "Severity classification: %s\n", ...
    severity);

fclose(fid);

%% Print Summary to Command Window

fprintf("\nCrack Segmentation Complete!\n");

fprintf( ...
    "Crack density: %.4f %%\n", ...
    crackDensity);

fprintf( ...
    "Mean crack width: %.4f pixels\n", ...
    meanCrackWidth_px);

fprintf( ...
    "Maximum crack width: %.4f pixels\n", ...
    maxCrackWidth_px);

fprintf( ...
    "Total crack length: %.2f pixels\n", ...
    totalCrackLength_px);

fprintf( ...
    "Longest continuous crack: %.2f pixels\n", ...
    longestCrack_px);

fprintf( ...
    "Number of connected cracks: %d\n", ...
    numCracks);

fprintf( ...
    "Severity classification: %s\n", ...
    severity);

fprintf( ...
    "Results saved in: %s\n", ...
    outputDir);