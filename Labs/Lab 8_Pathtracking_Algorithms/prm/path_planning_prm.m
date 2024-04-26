close all
clear
clc

load('map.mat')

ss = stateSpaceSE2;
ss.StateBounds = [map.XWorldLimits; map.YWorldLimits; [-pi pi]];

sv = validatorOccupancyMap(ss,Map=map);
sv.ValidationDistance = 0.01;

planner = plannerPRM(ss,sv);

graph = graphData(planner);

edges = table2array(graph.Edges);
nodes = table2array(graph.Nodes);
start = [0.1 2.4 0];
goal = [1.2 0.1 0];

show(sv.Map)
hold on
plot(nodes(:,1),nodes(:,2),"*","Color","b","LineWidth",2)
for i = 1:size(edges,1)
    % Samples states at distance 0.02 meters.
    states = interpolate(ss,nodes(edges(i,1),:), ...
                         nodes(edges(i,2),:),0:0.02:1);
    plot(states(:,1),states(:,2),"Color","b")
end
plot(start(1),start(2),"*","Color","g","LineWidth",3)
plot(goal(1),goal(2),"*","Color","r","LineWidth",3)

[pthObj, solnInfo] = plan(planner,start,goal);
pthObj.States

if solnInfo.IsPathFound
    interpolate(pthObj,1000);
    plot(pthObj.States(:,1),pthObj.States(:,2), ...
         "Color",[0.85 0.325 0.098],"LineWidth",2)
else
    disp("Path not found")
end
hold off