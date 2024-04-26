close all
clear
clc

load('map.mat')

planner = plannerAStarGrid(map);

init = [1,1];
goal = [49,25];

pathAstar = plan(planner,init,goal);

figure,
show(planner)