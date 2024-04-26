close all
clear
clc

load('map.mat')

ss = stateSpaceSE2;
ss.StateBounds = [map.XWorldLimits;map.YWorldLimits; [-pi pi]];

sv = validatorOccupancyMap(ss);
sv.Map = map;
sv.ValidationDistance = 0.01;

planner = plannerRRT(ss,sv);

start = [0.1 2.4 0];
goal = [1.2 0.1 0];

[pthObj,solnInfo] = plan(planner,start,goal);
pthObj.States

show(map)
hold on
plot(solnInfo.TreeData(:,1),solnInfo.TreeData(:,2),'.-'); % tree expansion
plot(pthObj.States(:,1),pthObj.States(:,2),'r-','LineWidth',2) % draw path