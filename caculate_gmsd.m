clear all,clc,close all
savepath='E:\data\after_spm\OUTPUT_REAL\GE\';
savepath_1='E:\data\after_spm\OUTPUT_REAL\XI\';
b = 0
for i=0:17
    im1=imread([savepath,num2str(i,'%d'),'','.jpg']);    %读入图片，如1_predict_prob.png
    %*********处理图片（省略）**********%
    if size(im1,3)==3
         im1=rgb2gray(im1);
    end
    im1 = imresize(im1,[256,256]);
    
    im2=imread([savepath_1,num2str(i,'%d'),'','.jpg']);    %读入图片，如1_predict_prob.png
    if size(im2,3)==3
         im2=rgb2gray(im2);
    end
    gg = GMSD(im1,im2);
    b = b + gg
    %im2=imread([num2str(i,'%d'),'','.jpg']);
end
b = b / 18