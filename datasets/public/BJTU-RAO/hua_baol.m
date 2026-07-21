function [OutputA,OutputF] = hua_baol(y,fs,style,varargin)
%画包络函数是hua_baoluo的代码简化版
%输入参数要么是3个，要么是5个
%当三个的时候
y_hht=hilbert(y);%希尔伯特变换
y_an=abs(y_hht);%包络信号
%y_an=y_an-mean(y_an);%去除直流分量
    if nargin==3
        [OutputA,OutputF] = hua_fft(y_an,fs,style);
    elseif nargin==5
        f1=varargin{1};
        f2=varargin{2};
        [OutputA,OutputF] = hua_fft(y_an,fs,style,f1,f2);
    else
        error('调用函数的输入参数数目不正确，输入参数只能是三个或者五个');
    end
end
