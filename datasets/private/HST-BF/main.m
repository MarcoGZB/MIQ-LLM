clear;clc;close all
%%
datafilefolder='D:\1.数据集\1.中心数据\ZXJB\fault_data\with_box\';
% =============== 样本选择 =============
type = "outer_pitting"; % outer_crack, outer_pitting, roller_crack, cage_crack  
load = 20;             % 0,20,40
num_id = 2;           % 0,1,2,3
channl = 7;           % 1~8: "idx","FSx","FSy","FSz","NSx","NSy","NSz","ch11"

% rpm = [1000,1500,2000];
rpm = [2000,2500,3000];
fs=12000;    % 采样频率
T = 1;       % 数据截取时长
len = fs*T;
order = ffCalculate(8.1, 77, 25, 10);  % 计算轴承故障特征频率(外、内、滚、保)

% 计算故障特征频率
ff = zeros(3,4);
for r = 1:length(rpm)
    ff(r,:) = rpm(r)*order/3.6/60;
end

switch type
    case "outer_crack"  %% 外圈裂纹
        position = ["Centered","Opposite","Orthogonal"];  position_id = position(2);
        outer_crack_fault= ["0.5","1","2"];
        N = 9;
        data_outer_crack =zeros(fs*T,N);
        for k = 1:length(rpm)        % 根据转速读取数据
            rpm_id  = rpm(k);
            for i = 1:length(outer_crack_fault)      % 根据故障程度读取数据
                fault_id=outer_crack_fault(i);
                datafilename = sprintf('%s_%smm_%s_%d_%d_%d.csv', type, fault_id, position_id, rpm_id, load, num_id);
                datafile = fullfile(datafilefolder, datafilename);
                raw_data = importdata(datafile);
                data_outer_crack(:,length(outer_crack_fault)*(k-1)+i) = raw_data.data(1:len, channl);
            end
        end
        data = data_outer_crack;
    case "outer_pitting" %% 外圈点蚀
        outer_pit_fault = ["light","moderate","severe"];
        N = 9;
        data_outer_pit =zeros(fs*T,N);
        for k = 1:length(rpm)        % 根据转速读取数据
            rpm_id  = rpm(k);
            for i = 1:length(outer_pit_fault)      % 根据故障程度读取数据
                outer_pit_id = outer_pit_fault(i);
                datafilename=sprintf('%s_%s_%d_%d_%d.csv', type, outer_pit_id, rpm_id, load, num_id); % 外圈点蚀
                datafile = fullfile(datafilefolder, datafilename);
                raw_data = importdata(datafile);
                data_outer_pit(:,length(outer_pit_fault)*(k-1)+i) = raw_data.data(1:len, channl);
            end
        end
        data = data_outer_pit;
    case "roller_crack"   %% 滚动体裂纹
        roller_cr_fault = ["0.4","0.8","1.2"];
        N = 9;
        data_roller_cr =zeros(fs*T,N);
        for k = 1:length(rpm)        % 根据转速读取数据
            rpm_id  = rpm(k);
            for i = 1:length(roller_cr_fault)      % 根据故障程度读取数据
                roller_cr_id = roller_cr_fault(i);
                datafilename=sprintf('%s_%smm_%d_%d_%d.csv', type, roller_cr_id, rpm_id, load, num_id); % 滚动体裂纹
                datafile = fullfile(datafilefolder, datafilename);
                raw_data = importdata(datafile);
                data_roller_cr(:,length(roller_cr_fault)*(k-1)+i) = raw_data.data(1:len, channl);
            end
        end
        data = data_roller_cr;
    case "cage_crack"   %% 保持架裂纹
        N = 3;
        data_cage_cr =zeros(fs*T,N);
        for k = 1:length(rpm)        % 根据转速读取数据
            rpm_id  = rpm(k);
            datafilename=sprintf('%s_%d_%d_%d.csv', type, rpm_id, load, num_id); % 滚动体裂纹
            datafile = fullfile(datafilefolder, datafilename);
            raw_data = importdata(datafile);
            data_cage_cr(:,k) = raw_data.data(1:len, channl);
        end
        data = data_cage_cr;
end

axle_freq = rpm/60/3.6;
fault_freq = axle_freq'.*order;

disp(['转频：', num2str(axle_freq)]);
for r = 1:length(rpm)
    disp([num2str(rpm(r)),'故障特征频率：', num2str(fault_freq(r,:))]);
end

t=0:1/fs:(T-1/fs);  % 采样时间序列
nfft= 2^nextpow2(length(data));
idx = 0:round(nfft/2-1);   % 频谱的点数
k = idx*fs/nfft;           % 频谱的频率序列

%% 画时域图及其频谱、包络谱
% ffts = zeros(nfft,N);
% xl1 = 1000;           
% for i=1:3  % 三种转速
%     for j = 1:3  % 每种转速下三种故障程度
%     figure
%     subplot(3,1,1); plot(t(1:len),data(:,3*(i-1)+j));set(gca,'FontName','Times New Roman','FontSize',12);
% %     xlabel('\fontname{华文中宋}时间\fontname{Times new roman}(s)');
%     ylabel('\fontname{华文中宋}幅值\fontname{Times new roman}(m/s²)');
%     
%     ffts(:,3*(i-1)+j)=fft(data(:,3*(i-1)+j),nfft)*2/nfft;
%     subplot(3,1,2); plot(k,abs(ffts(idx+1,3*(i-1)+j))); set(gca,'FontName','Times New Roman','FontSize',12);
% %     xlabel('\fontname{华文中宋}频率\fontname{Times new roman}(Hz)');
%     ylabel('\fontname{华文中宋}幅值\fontname{Times new roman}(m/s²)');
% 
%     subplot(3,1,3);hua_baol(data(:,3*(i-1)+j),fs,1,0,xl1);
%     set(gca,'FontName','Times New Roman','FontSize',12);
% %     xlabel('\fontname{华文中宋}频率\fontname{Times new roman}(Hz)');
%     ylabel('\fontname{华文中宋}幅值\fontname{Times new roman}(m/s²)');
%     set (gcf,'Position',[100+700*(j-1),800-400*(i-1),600,300]);
%     end
% end


ffts = zeros(nfft,N);
xl1 = 500;           
for i=1:3  % 三种转速
    for j = 1:3  % 每种转速下三种故障程度
    figure
    hua_baol(data(:,3*(i-1)+j),fs,1,0,xl1);
    set(gca,'FontName','Times New Roman','FontSize',12);
%     xlabel('\fontname{华文中宋}频率\fontname{Times new roman}(Hz)');
    ylabel('\fontname{华文中宋}幅值\fontname{Times new roman}(m/s²)');
    set (gcf,'Position',[100+650*(j-1),800-300*(i-1),600,200]);
    hold on;
    yl = ylim; % 获取y轴的范围
    plot([axle_freq(i) axle_freq(i)], [yl(2) 1.5*mean(ylim)], '--r', 'LineWidth', 0.8);
    hold off;
    end
end

% c = zeros(fs*T,N);
% for i = 1:9
%     [c,f,S] = demo_Fast_Kurtogram(data(:,i),fs,2,1,2);
%     nfft_c = 2^nextpow2(length(c));
%     figure
%     plot(f,S(1:nfft_c/2));
%     set (gcf,'Position',[50+150*(i-1),400,500,100]);
% end





