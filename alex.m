
% Sara Bagherzadeh, Biomedical Engineering P.h.D student at Science and
% Research Branch, Islamic Azad University, Tehran, Iran. Email:
% sara.bagherzadeh95@gmail.com or sara.bagherzadeh@srbiau.ac.ir .                                  
%project name=  recognition of emotion from EEG signals using pre-trained CNN
%models and brain effective connectivity measure the direct directed transfer function (dDTF)


% We have a folder named 'dDTF' which includes 5 subfolders of 'q1', 'q2',
% 'q3', 'q4' and 'neu' according two five classes of emotional states. Each
% subfolder contains ddtf images; dDTF brain connectivity was estimated using the 
%'sift toolbox'. You can install and learn how to estimate EC measures such as dDTF from bellow link:
%https://sccn.ucsd.edu/wiki/SIFT
%(Also, first, EEG signals were pre-processed using EEGLAB toolbox at MATLAB software environment,
%You can install and learn how to pre-process EEG signals using EEGLAB from bellow link:
%https://sccn.ucsd.edu/eeglab/index.php)
% After estimate dDTF, plot and save images in the 5 mentioned subfolders now we can use pre-trained
% CNN models such as AlexNet, ResNet-50, Inception-v3 and VGG-19. Here we get codes for fine-tune and 
% classify using AlexNet for our dDTF images.    
% this code is released on 11 November 2020. 


%load the new images as an image datastore. Divide the data into 90% training data and 10% validation data.

imds = imageDatastore('dDTF','IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.9,'randomized');

net = alexnet;
analyzeNetwork(net)

inputSize = net.Layers(1).InputSize
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels))

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];


%Resize images to match the pretrapined network input size.
augimdsTrain = augmentedImageDatastore([227 227],imdsTrain);
augimdsValidation = augmentedImageDatastore([227 227],imdsValidation);


options = trainingOptions('adam', ...
    'InitialLearnRate',1e-4, ...
    'SquaredGradientDecayFactor',0.99, ...
    'MaxEpochs',40, ...
    'MiniBatchSize',64, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain,layers,options);

[YPred,scores] = classify(netTransfer,augimdsValidation);
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)

group=imdsValidation.Labels;
grouphat=YPred;
plotconfusion(group,grouphat)
