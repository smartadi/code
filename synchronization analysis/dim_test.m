clc;
clear all;close all;


im = zeros(200,300);

im(150,250) = 1;



figure()
imshow(im);hold on;
plot(250,150,'bo');
axis on