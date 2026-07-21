function FFTplot(signal, Fs,color)
    % FFTplot - 绘制信号的频谱图
    %
    % 输入:
    %   signal - 要分析的时间序列信号
    %   Fs - 信号的采样频率

    % 计算信号长度
    L = length(signal); 

    % 计算FFT
    Y = fft(signal); 
    P2 = abs(Y/L); 
    P1 = P2(1:L/2+1); 

    % 计算频率轴的值
    f = Fs*(0:(L/2))/L; 

    % 绘制频谱图
    plot(f, P1,color = color,LineWidth=1.1) 
    xlabel('Frequency ($f$)',Interpreter='latex')
    ylabel('$|P1(f)|$',Interpreter='latex')
    ylim([0 0.05])
end