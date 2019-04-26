clear all

% Load the data
data = spinningtop('spinningtopbest.txt');

% Vectorise coordinate data
t = table2array(data(:, 1));
psi = table2array(data(:,2));
phi = table2array(data(:,3));
theta = table2array(data(:,4));

% Radius of sphere
r = 0.7;

% Convert to cartesian
X = r .* sin(theta) .* cos(phi);
Y = r .* sin(theta) .* sin(phi);
Z = r .* cos(theta);

% Setup graph
figure;

% Draw sphere radius r
[x,y,z] = sphere;

x = x * r
y = y * r
z = z * r

s = surf(x,y,z);
hold on;

% Plot our data on top of sphere
p = plot3(X,Y,Z,'k');
p.LineWidth = 2;

% Hide grid
grid off
set(gca,'visible','off')