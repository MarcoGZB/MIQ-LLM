function [EN_Description] = ENDescription(TimeHI,FreqHI,conditionlabel,fs,L)
FeatureCell_T = {'Mean value';'Standard deviation';'Square root amplitude';'Absolute mean value';'Peak value';'Skewness';
               'Kurtosis';'Variance';'Kurtosis index';'Peak index';'Waveform index';'Pulse index'};
FeatureCell_F = {'Frequency mean value';'Frequency variance';'Frequency skewness';'Frequency kurtosis';'Gravity frequency';'Frequency standard deviation';
               'Frequency root mean square';'Average frequency';'Regularity degree';'Variation parameter';'Eighth-order moment';'Sixteenth-order moment'};

EN_Description1 = 'You are an engineer working on fault diagnosis based on digital signal processing, with a focus on bearings, and you need to diagnose a vibration signal based on a section of the vibration signal and its time-frequency domain characteristics.';
EN_Description2_Condition = 'The operating conditions of this vibration signal are as follows:';

switch conditionlabel
    case 0
        Descondition_EN = 'Speed:1500rpm; Torque:0.7Nm; Load:1000N';
    case 1
        Descondition_EN = 'Speed:900rpm; Torque:0.7Nm; Load:1000N';
    case 2
        Descondition_EN = 'Speed:1500rpm; Torque:0.1Nm; Load:1000N';
    case 3
        Descondition_EN = 'Speed:1500rpm; Torque:0.7Nm; Load:400N';
end
EN_conditionDes = strcat(EN_Description2_Condition,Descondition_EN,',');
EN_Description2_T = 'Time domain characteristics of this signal:';
EN_Description2_F = 'Frequency domain characteristics of this signal:';
DesEnd = ['This signal data is as follows and its sampling frequency fs is:' num2str(fs) 'Hz，Sampling time:' num2str(L/fs,4) 's,Please diagnose the type of fault in the signal based on its time-frequency domain characteristics and raw data.'];
FeaDes_T = EN_Description2_T;
FeaDes_F = EN_Description2_F;
for i = 1:12
    FeaDes_T = strcat('The value of' ,FeaDes_T,FeatureCell_T{i},'is:' ,num2str(TimeHI(i),4),';');
    FeaDes_F = strcat('The value of' ,FeaDes_F,FeatureCell_F{i},'is:' ,num2str(FreqHI(i),4),';');
end

EN_Description = strcat(EN_Description1, ...
    EN_conditionDes, ...
    FeaDes_T,FeaDes_F, ...
    DesEnd);
