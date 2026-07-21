function [TimeHI,FreqHI]=HIComputation(SelectedSceq,fs)
TimeHI = [ calculate_p1(SelectedSceq)%Mean Value
           calculate_p2(SelectedSceq)%Standard deviation
           calculate_p3(SelectedSceq)%Square root amplitude
           calculate_p4(SelectedSceq)%Absolute mean value
           calculate_p5(SelectedSceq)%Peak value
           calculate_p6(SelectedSceq)%Skewness
           calculate_p7(SelectedSceq)%Kurtosis
           calculate_p8(SelectedSceq)%Variance
           calculate_p9(SelectedSceq)%Kurtosis index
          calculate_p10(SelectedSceq)%Peak index
          calculate_p11(SelectedSceq)%Waveform index
          calculate_p12(SelectedSceq)%Pulse index
          ];
FreqHI = [calculate_p13(SelectedSceq,fs)%Frequency mean value
          calculate_p14(SelectedSceq,fs)%Frequency variance
          calculate_p15(SelectedSceq,fs)%Frequency skewness
          calculate_p16(SelectedSceq,fs)%Frequency kurtosis
          calculate_p17(SelectedSceq,fs)%Gravity frequency
          calculate_p18(SelectedSceq,fs)%Frequency standard deviation
          calculate_p19(SelectedSceq,fs)%Frequency root mean square
          calculate_p20(SelectedSceq,fs)%Average frequency
          calculate_p21(SelectedSceq,fs)%Regularity degree
          calculate_p22(SelectedSceq,fs)%Variation parameter
          calculate_p23(SelectedSceq,fs)%Eighth-order moment
          calculate_p24(SelectedSceq,fs)%Sixteenth-order moment
          ];
end

%% 时域指标函数
function p1 = calculate_p1(x)
    p1 = mean(x);
end

function p2 = calculate_p2(x)
    p2 = std(x, 1); % 样本标准差（分母N-1）
end

function p3 = calculate_p3(x)
    p3 = (mean(sqrt(abs(x))))^2;
end

function p4 = calculate_p4(x)
    p4 = mean(abs(x));
end

function p5 = calculate_p5(x)
    p5 = max(abs(x));
end

function p6 = calculate_p6(x)
    p6 = mean(x.^3);
end

function p7 = calculate_p7(x)
    p7 = mean(x.^4);
end

function p8 = calculate_p8(x)
    p8 = mean(x.^2);
end

function p9 = calculate_p9(x)
    p7 = calculate_p7(x);
    p6 = calculate_p6(x);
    p9 = p7 / (sqrt(p6))^2;
end

function p10 = calculate_p10(x)
    p5 = calculate_p5(x);
    p2 = calculate_p2(x);
    p10 = p5 / p2;
end

function p11 = calculate_p11(x)
    p2 = calculate_p2(x);
    p4 = calculate_p4(x);
    p11 = p2 / p4;
end

function p12 = calculate_p12(x)
    p5 = calculate_p5(x);
    p4 = calculate_p4(x);
    p12 = p5 / p4;
end

%% 频域指标函数（需先计算频谱）
function [s, f] = compute_spectrum(x, fs)
    N = length(x);
    %去均值
    x = x-mean(x);
    %希尔伯特变换
    Hx = hilbert(x);
    f1 = abs(Hx);
    f2 = f1 - mean(f1);
    s = 2*abs(fft(f2)/N); % 单边幅度谱
    s = s(1:floor(N/2)+1);
    f = (0:floor(N/2)) * fs / N;
end

function p13 = calculate_p13(x, fs)
    [s, ~] = compute_spectrum(x, fs);
    p13 = mean(s);
end

function p14 = calculate_p14(x, fs)
    [s, ~] = compute_spectrum(x, fs);
    p14 = var(s, 1); % 样本方差（分母K-1）
end

function p15 = calculate_p15(x, fs)
    [s, ~] = compute_spectrum(x, fs);
    p13 = calculate_p13(x, fs);
    p14 = calculate_p14(x, fs);
    p15 = mean((s - p13).^3) / (p14^(3/2));
end

function p16 = calculate_p16(x, fs)
    [s, ~] = compute_spectrum(x, fs);
    p13 = calculate_p13(x, fs);
    p14 = calculate_p14(x, fs);
    p16 = mean((s - p13).^4) / (p14^2);
end

function p17 = calculate_p17(x, fs)
    [s, f] = compute_spectrum(x, fs);
    p17 = sum(f .* s) / sum(s);
end

function p18 = calculate_p18(x, fs)
    [s, f] = compute_spectrum(x, fs);
    p17 = calculate_p17(x, fs);
    K = length(s);
    p18 = sqrt(sum((f - p17).^2 .* s) / (K * sum(s)));
end

function p19 = calculate_p19(x, fs)
    [s, f] = compute_spectrum(x, fs);
    p19 = sqrt(sum(f.^2 .* s) / sum(s));
end

function p20 = calculate_p20(x, fs)
    [s, f] = compute_spectrum(x, fs);
    numerator = sum(f.^4 .* s);
    denominator = sum(f.^2 .* s);
    p20 = sqrt(numerator / denominator);
end

function p21 = calculate_p21(x, fs)
    [s, f] = compute_spectrum(x, fs);
    
    numerator = sum(f.^2 .* s);
    denominator = sqrt(sum(s)/sum(f.^4 .* s));
    p21 = numerator / denominator;
end

function p22 = calculate_p22(x, fs)
    p18 = calculate_p18(x, fs);
    p17 = calculate_p17(x, fs);
    p22 = p18 / p17;
end

function p23 = calculate_p23(x, fs)
    [s, f] = compute_spectrum(x, fs);
    p17 = calculate_p17(x, fs);
    p18 = calculate_p18(x, fs);
    K = length(s);
    p23 = sum((f - p17).^3 .* s) / (K * p18^3);
end

function p24 = calculate_p24(x, fs)
    [s, f] = compute_spectrum(x, fs);
    p17 = calculate_p17(x, fs);
    p18 = calculate_p18(x, fs);
    K = length(s);
    p24 = sum((f - p17).^4 .* s) / (K * p18^4);
end