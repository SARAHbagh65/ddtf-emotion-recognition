
% Sara Bagherzadeh, Biomedical Engineering P.h.D student at Science and
% Research Branch, Islamic Azad University, Tehran, Iran. Email:
% sara.bagherzadeh95@gmail.com or sara.bagherzadeh@srbiau.ac.ir .                                  
%project name=  recognition of emotion from EEG signals using pre-trained CNN
%models and brain effective connectivity measure the direct directed transfer function (dDTF)


% We have a folder named 'dDTF' which includes 5 subfolders of 'q1', 'q2',
% 'q3', 'q4' and 'neu' representing each of five emotional states. Each
% subfolder contains ddtf images; dDTF brain connectivity was estimated using the 
%'sift toolbox'. You can install and learn how to estimate EC measures such as dDTF from bellow link:
%https://sccn.ucsd.edu/wiki/SIFT
%(Also, first, EEG signals were pre-processed using EEGLAB toolbox in MATLAB software environment,
%You can install and learn how to pre-process EEG signals using EEGLAB from bellow link:
%https://sccn.ucsd.edu/eeglab/index.php)
% After estimating dDTF, plot and save images in the 5 mentioned subfolders now we can use pre-trained
% CNN models such as AlexNet, ResNet-50, Inception-v3 and VGG-19. Here we get codes for fine-tune and 
% classify using ResNet-50 for our dDTF images.    
% this code is released on 14 November 2020. 


%load the new images as an image datastore. Divide the data into 90% training data and 10% validation data.
imds = imageDatastore('te','IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.9,'randomized');

%Resize images to match the pretrapined network input size.
augimdsTrain = augmentedImageDatastore([224 224],imdsTrain);
augimdsValidation = augmentedImageDatastore([224 224],imdsValidation);
% if you want use inception-v3 or xception, change size of input image in two upper lines to [299 299]. 

options = trainingOptions('adam', ...
    'InitialLearnRate',4e-4, ...
    'SquaredGradientDecayFactor',0.99, ...
    'MaxEpochs',40, ...
    'MiniBatchSize',64, ...
    'Plots','training-progress');

net=resnet50;
% as you want use resnet18, shufflenet, inception-v3 or xception, just write its name isttead of 'resnet50' 
%in upper line.

% recall deep learning toolbox
deepNetworkDesigner;
% after opening the 'deepNetworkDesigner' window, import resnet18 and replace fully connected layer 
%with a new one containing 5 neurons (5 emotional classes), modify "WeightLearnRateFactor" and
% "BiasLearnRateFactor" and set both of them 20. Then replace
% classification layer with a new one and connect wirings. Now export the results as 'lgraph_1'.  

%load lgraph_1.mat
%To train the network, supply the layers exported from the app, layers_1, the training images, and options, to the trainNetwork function. By default, trainNetwork uses a GPU if available (requires Parallel Computing Toolbox™). Otherwise, it uses a CPU. Training is fast because the data set is so small.
trainedNet = trainNetwork(augimdsTrain,lgraph_1,options);
[YPred,probs] = classify(trainedNet,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)

