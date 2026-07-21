function [CN_Description] = CNDescription(TimeHI,FreqHI,conditionlabel,fs,L)
FeatureCell_T = {'Mean value';'Standard deviation';'Square root amplitude';'Absolute mean value';'Peak value';'Skewness';
               'Kurtosis';'Variance';'Kurtosis index';'Peak index';'Waveform index';'Pulse index'};
FeatureCell_F = {'Frequency mean value';'Frequency variance';'Frequency skewness';'Frequency kurtosis';'Gravity frequency';'Frequency standard deviation';
               'Frequency root mean square';'Average frequency';'Regularity degree';'Variation parameter';'Eighth-order moment';'Sixteenth-order moment'};

CN_Description1 = '你是一个基于数字信号处理进行故障诊断的工程师，主要研究对象轴承，你需要根据一段振动信号和其时频域特征进行诊断,';
CN_Description2_Condition = '该振动信号的运行工况如下：';

switch conditionlabel
    case 0
        Descondition_CN = '转速：1500rpm；扭矩：0.7Nm；载荷：1000N';
    case 1
        Descondition_CN = '转速：900rpm；扭矩：0.7Nm；载荷：1000N';
    case 2
        Descondition_CN = '转速：1500rpm；扭矩：0.1Nm；载荷：1000N';
    case 3
        Descondition_CN = '转速：1500rpm；扭矩：0.7Nm；载荷：400N';
end
CN_conditionDes = strcat(CN_Description2_Condition,Descondition_CN,',');
CN_Description2_T = '该信号时域特征';
CN_Description2_F = '该信号频域特征';
DesEnd = ['这段信号数据如下，其采样频率fs为:' num2str(fs) 'Hz，采样时间T为:' num2str(L/fs,4) 's,请你根据信号时频域特征和原始数据诊断其故障类型。'];
FeaDes_T = CN_Description2_T;
FeaDes_F = CN_Description2_F;
for i = 1:12
    FeaDes_T = strcat(FeaDes_T,FeatureCell_T{i},'的值为：' ,num2str(TimeHI(i),4),'；');
    FeaDes_F = strcat(FeaDes_F,FeatureCell_F{i},'的值为：' ,num2str(FreqHI(i),4),'；');
end
CN_Description = strcat(CN_Description1, ...
    CN_conditionDes, ...
    FeaDes_T,FeaDes_F, ...
    DesEnd);
end


