function [DamageDes, Type_Code] = TargetDescription(BT,BearingType,DamageType)

loc = find(strcmp(BearingType, BT));
DamageDes = ['The fault type is: ', DamageType{loc},'.'];
if loc <= 8
    Type_Code = 1;
elseif loc >= 17
    Type_Code = 0;
else
    Type_Code = 2;

end