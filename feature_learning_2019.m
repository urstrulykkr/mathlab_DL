clear all;
close all;
tic;
imds = imageDatastore('I:\Imp\TL\Alexnet\warwick','IncludeSubfolders',true,'LabelSource','foldernames');
imds.ReadFcn = @readFunctionTrain;
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');
%net = alexnet;
%net = vgg16;
net = inceptionv3;
inputSize = net.Layers(1).InputSize;
analyzeNetwork(net)
%----Extract FEatures-----
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);
%layer = 'fc8';%--- Alexnet----
layer = 'predictions';%---inception
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;
tim = toc;
classifier = fitcecoc(featuresTrain,YTrain);
YPred = predict(classifier,featuresTest);
YValid= imdsTest.Labels;
yp = grp2idx(YPred);
yv = grp2idx(YValid);
[c_matrix,Result]= confusion_cal.getMatrix(yp,yv);
tim