function Data = PardFileRead(path,num,type)
    % PardFileRead - 读取帕德博恩轴承数据集信号
    %
    % 输入:
    %   file - 文件路径
    %   num - 数据组号
    %   type - 数据类别
    %       振动：'Vib'
    %       转速：'Spe'
    %       扭矩：'Tor'
    %       外力：'For'

    %数据组号
    Num = num;%
    file = [path '_' num2str(Num) '.mat'];
    data = importdata(file);
    data1 = data.Y;
    
    if type == 'Vib'
        %vibration
        Data = data1(7).Data;
    elseif type == 'Spe'
        %Speed
        Data = data1(4).Data;
    elseif type == 'Tor'
        %Torque
        Data = data1(6).Data;
    elseif type == 'force'
        %force
        Data = data1(6).Data;
    end
end