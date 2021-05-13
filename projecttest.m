clc; 
close all; 
clear all;
% Specify the folder where normal samples are.
myFolder = 'C:\Users\Muskaan Patel\Desktop\Research Paper\Images\skin-cancer-malignant-vs-benign\train\benign';
% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, '*.jpg'); % Samples images are bmp format. Modify it if necessary.
theFiles = dir(filePattern);
% Initialize matrix of normal features
n = length(theFiles);
normal_feature = zeros(n,5); 
%normal_feature(:,5) = zeros(n,end);
for i = 1 : n
  baseFileName = theFiles(i).name;
  fullFileName = fullfile(myFolder, baseFileName);
  RGB_sample = imread(fullFileName); % Read RGB image.
  RGB_sample=im2double(RGB_sample);
  h=fspecial('laplacian',0.1);
  RGB1=imfilter(RGB_sample,h);%Filtering
  gray_sample = rgb2gray(RGB_sample); % Convert RGB image to gray scale. 
  NormalSample = graythresh(gray_sample);
  s=strel('disk',2,0);
  F=imerode(gray_sample,s); %Erosion for BoundaryExtraction
  F=im2double(F);
  G=medfilt2(F); %MedianFilter
  H=histeq(G); %Histogram Equalisin
  K=im2bw(gray_sample,0.5);
  BoundaryExtracted=edge(K);
  %FeatureExtraction
  hello = regionprops('table',BoundaryExtracted,'MajorAxisLength','MinorAxisLength','ConvexArea','Solidity','Circularity');
%   hello=table2array(hello);
%   hello=hello';
%   normal_feature(i,1)=max(hello(1,:));
%   normal_feature(i,2)=max(hello(2,:));
%   normal_feature(i,3)=max(hello(3,:));
%   normal_feature(i,4)=max(hello(4,:));
%   normal_feature(i,5)=max(hello(5,:));
end
 
 
% Specify the folder where cancer samples are.
myFolder1 = 'C:\Users\Muskaan Patel\Desktop\Research Paper\Images\skin-cancer-malignant-vs-benign\train\malignant';
% Get a list of all files in the folder with the desired file name pattern.
filePattern1 = fullfile(myFolder1, '*.jpg'); % Samples images are bmp format. Modify it if necessary.
theFiles1 = dir(filePattern1);
% Initialize matrix of normal features
m = length(theFiles1);
cancer_feature = zeros(m,5); 
%cancer_feature(:,5) = zeros(m,end);
for j = 1 : m
  baseFileName1 = theFiles1(j).name;
  fullFileName1 = fullfile(myFolder1, baseFileName1);
  RGB_sample1 = imread(fullFileName1); % Read RGB image.
  RGB_sample1=im2double(RGB_sample1);
  h=fspecial('laplacian',0.1);
  RGB2=imfilter(RGB_sample1,h);%Filtering
  gray_sample1 = rgb2gray(RGB_sample1); % Convert RGB image to gray scale. 
  CancerSample = graythresh(gray_sample1);
  s=strel('disk',2,0);
  f=imerode(gray_sample1,s); %Erosion for BoundaryExtraction
  f=im2double(f);
  g=medfilt2(f); %MedianFilter
  h=histeq(g); %Histogram Equalisin
  k=im2bw(gray_sample1,0.5);
  BoundaryExtracted1=edge(k);
  %FeatureExtraction
  hello1 = regionprops('table',BoundaryExtracted1,'MajorAxisLength','MinorAxisLength','ConvexArea','Solidity','Circularity'); %FEATURE 1
  hello1=table2array(hello1);
  hello1=hello1';
  cancer_feature(j,1)=max(hello1(1,:));
  cancer_feature(j,2)=max(hello1(2,:));
  cancer_feature(j,3)=max(hello1(3,:));
  cancer_feature(j,4)=max(hello1(4,:));
  cancer_feature(j,5)=max(hello1(5,:));
end

features = [normal_feature; cancer_feature];
[linear_corr,pval] = corr(features(:,1:5));

n = size(normal_feature,1); 
c = size(cancer_feature,1);
MajorAxisLength = vertcat(normal_feature(:,1),cancer_feature(:,1));
MinorAxisLength = vertcat(normal_feature(:,2),cancer_feature(:,2));
ConvexArea = vertcat(normal_feature(:,3),cancer_feature(:,3));
Solidity = vertcat(normal_feature(:,4),cancer_feature(:,4));
Circularity = vertcat(normal_feature(:,5),cancer_feature(:,5));
Ground_truth = vertcat(repmat(['Normal'],n,1),repmat(['Cancer'],c,1));
data = table(MajorAxisLength,MinorAxisLength,ConvexArea,Solidity,Circularity,Ground_truth);






% Specify the folder where normal samples are.
myFoldertest = 'C:\Users\Muskaan Patel\Desktop\Research Paper\Images\skin-cancer-malignant-vs-benign\test\benign';
% Get a list of all files in the folder with the desired file name pattern.
filePatterntest = fullfile(myFoldertest, '*.jpg'); % Samples images are bmp format. Modify it if necessary.
theFilestest = dir(filePatterntest);
% Initialize matrix of normal features
ntest = length(theFilestest);
normal_feature_test = zeros(ntest,5); 
%normal_feature(:,5) = zeros(n,end);
for itest = 1 : ntest
  baseFileNameTest = theFilestest(itest).name;
  fullFileNameTest = fullfile(myFoldertest, baseFileNameTest);
  RGB_sampleTest = imread(fullFileNameTest); % Read RGB image.
  RGB_sampleTest = im2double(RGB_sampleTest);
  h=fspecial('laplacian',0.1);
  RGB1test=imfilter(RGB_sampleTest,h);%Filtering
  gray_sampleTest = rgb2gray(RGB_sampleTest); % Convert RGB image to gray scale. 
  NormalSampleTest = graythresh(gray_sampleTest);
  s=strel('disk',2,0);
  Ftest=imerode(gray_sampleTest,s); %Erosion for BoundaryExtraction
  Ftest=im2double(Ftest);
  Gtest=medfilt2(Ftest); %MedianFilter
  Htest=histeq(Gtest); %Histogram Equalisin
  Ktest=im2bw(gray_sampleTest,0.5);
  BoundaryExtractedTest=edge(Ktest);
  %FeatureExtraction
  helloTest = regionprops('table',BoundaryExtractedTest,'MajorAxisLength','MinorAxisLength','ConvexArea','Solidity','Circularity'); 
  helloTest=table2array(helloTest);
  helloTest=helloTest';
  normal_feature_test(itest,1)=max(helloTest(1,:));
  normal_feature_test(itest,2)=max(helloTest(2,:));
  normal_feature_test(itest,3)=max(helloTest(3,:));
  normal_feature_test(itest,4)=max(helloTest(4,:));
  normal_feature_test(itest,5)=max(helloTest(5,:));
end
 
 
% Specify the folder where cancer samples are.
myFolder1test = 'C:\Users\Muskaan Patel\Desktop\Research Paper\Images\skin-cancer-malignant-vs-benign\test\malignant';
% Get a list of all files in the folder with the desired file name pattern.
filePattern1test = fullfile(myFolder1test, '*.jpg'); % Samples images are bmp format. Modify it if necessary.
theFiles1test = dir(filePattern1test);
% Initialize matrix of normal features
mtest = length(theFiles1test);
cancer_feature_test = zeros(mtest,5); 
%cancer_feature(:,5) = zeros(m,end);
for jtest = 1 : mtest
  baseFileName1test = theFiles1test(jtest).name;
  fullFileName1test = fullfile(myFolder1test, baseFileName1test);
  RGB_sample1test = imread(fullFileName1test); % Read RGB image.
  RGB_sample1test=im2double(RGB_sample1test);
  h=fspecial('laplacian',0.1);
  RGB2test=imfilter(RGB_sample1test,h);%Filtering
  gray_sample1test = rgb2gray(RGB_sample1test); % Convert RGB image to gray scale. 
  CancerSampletest = graythresh(gray_sample1test);
  s=strel('disk',2,0);
  ftest=imerode(gray_sample1test,s); %Erosion for BoundaryExtraction
  ftest=im2double(ftest);
  gtest=medfilt2(ftest); %MedianFilter
  htest=histeq(gtest); %Histogram Equalisin
  ktest=im2bw(gray_sample1test,0.5);
  BoundaryExtracted1test=edge(ktest);
  %FeatureExtraction
  hello1test = regionprops('table',BoundaryExtracted1test,'MajorAxisLength','MinorAxisLength','ConvexArea','Solidity','Circularity'); %FEATURE 1
  hello1test=table2array(hello1test);
  hello1test=hello1test';
  cancer_feature_test(jtest,1)=max(hello1test(1,:));
  cancer_feature_test(jtest,2)=max(hello1test(2,:));
  cancer_feature_test(jtest,3)=max(hello1test(3,:));
  cancer_feature_test(jtest,4)=max(hello1test(4,:));
  cancer_feature_test(jtest,5)=max(hello1test(5,:));
end

featuresTest = [normal_feature_test; cancer_feature_test];
[linear_corr,pval] = corr(featuresTest(:,1:5));

ntest = size(normal_feature_test,1); 
ctest = size(cancer_feature_test,1);
MajorAxisLengthTest = vertcat(normal_feature_test(:,1),cancer_feature_test(:,1));
MinorAxisLengthTest = vertcat(normal_feature_test(:,2),cancer_feature_test(:,2));
ConvexAreaTest = vertcat(normal_feature_test(:,3),cancer_feature_test(:,3));
SolidityTest = vertcat(normal_feature_test(:,4),cancer_feature_test(:,4));
CircularityTest = vertcat(normal_feature_test(:,5),cancer_feature_test(:,5));
Ground_truthTest = vertcat(repmat(['Normal'],n,1),repmat(['Cancer'],c,1));
dataTest = table(MajorAxisLengthTest,MinorAxisLengthTest,ConvexAreaTest,SolidityTest,CircularityTest,Ground_truthTest);
train=table2array(data);
test=table2array(dataTest);

for i= 1: width(data)
    data.(i)(isnan(data.(i))|isinf(data.(i))) = 0;
end    
for i= 1: width(dataTest)
    dataTest.(i)(isnan(dataTest.(i))|isinf(dataTest.(i))) = 0;
end    