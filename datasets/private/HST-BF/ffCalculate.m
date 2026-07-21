function order = ffCalculate(Bd,BD,BZ,Ba) %滚子直径，轴承节径，滚动体个数，接触角
    order_o=0.5*BZ*(1-Bd*cosd(Ba)/BD);  % 外圈
    order_i=0.5*BZ*(1+Bd*cosd(Ba)/BD);  % 内圈
    order_b=0.5*(BD/Bd)*(1-(Bd/BD)^2*cosd(Ba)^2);  % 滚动体
    order_c=0.5*(1-Bd*(cosd(Ba))/BD);  % 保持架
    order = [order_o, order_i, order_b, order_c];
end