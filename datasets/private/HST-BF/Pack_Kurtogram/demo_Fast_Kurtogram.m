% clc;clear
% load('130.mat')
function [c,f,S] = demo_Fast_Kurtogram(x,Fs,nlevel,opt1,opt2)
[c,f,S] = Fast_Kurtogram(x,nlevel,Fs,opt1,opt2);
% % Pre-whitening the signal (optional)
% % prewith = input('Do you want to prewhiten the signal ? (no = 0 ; yes = 1): ');
% % if prewith == 1
%    x = x - mean(x);
%    Na = 100;
%    a = lpc(x,Na);
%    x = fftfilt(a,x);
%    x = x(Na+1:end);   % it is very important to remove the transient of the whitening filter, otherwise the SK will detect it!!
% % end
% for i = 1:P
%   c(i,:) = Fast_Kurtogram(x(:,i),nlevel,Fs,opt1,opt2);
% end

