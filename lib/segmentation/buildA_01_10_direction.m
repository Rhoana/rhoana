%this function builds connectivity matrix A and T for E(x,y) for GraphCut
%ASSUMPTIONS: E00 == E11 == 0 and E01 != E10
%[A,T] = buildA_01_10_for3(index, im, lambda, ns, smoothFunc, params)
function [A,T] = buildA_01_10_direction_fast(index, im,lambda, ns, smoothFunc, params)
n = numel(im);
h = size(im,1);
A = sparse(n,n);
T = sparse(n,2);

smoothFunc01 = strcat(smoothFunc,'01');
smoothFunc10 = strcat(smoothFunc,'10');

%#function E_direction01
%#function E_direction10

%links in im to neighbouring pixels
i = 1:n-ns*h; % right
j = i+ns*h;
[orientImg, distance] = buildOrientationImage(im,i(1),j(1),params);
params.orientImg = orientImg;
params.dist = distance;
T = T + sparse(i,ones(length(i),1), feval(smoothFunc10,im,i,j,params),n, 2);
T = T + sparse(j,ones(length(i),1)*2, feval(smoothFunc01,im,i,j,params),n, 2);
A = A + lambda*sparse(i,j,feval(smoothFunc10,im,i,j,params)+feval(smoothFunc01,im,i,j,params),n,n);

i = ns*h+1:n; %left
j = i-ns*h;
[orientImg, distance] = buildOrientationImage(im,i(1),j(1),params);
params.orientImg = orientImg;
params.dist = distance;
T = T + sparse(i,ones(length(i),1), feval(smoothFunc10,im,i,j,params),n,2);
T = T + sparse(j,ones(length(i),1)*2, feval(smoothFunc01,im,i,j,params),n,2);
A = A + lambda*sparse(i,j,feval(smoothFunc10,im,i,j,params)+feval(smoothFunc01,im,i,j,params),n,n);

i = 1:n;
i(find(mod(i,h) == 0)) = []; %down
for k=1:ns-1
    i(find(mod(i,h) == h-k)) = []; %down
end
j = i+ns;
[orientImg, distance] = buildOrientationImage(im,i(1),j(1),params);
params.orientImg = orientImg;
params.dist = distance;
T = T + sparse(i,ones(length(i),1), feval(smoothFunc10,im,i,j,params),n, 2);
T = T + sparse(j,ones(length(i),1)*2, feval(smoothFunc01,im,i,j,params),n, 2);
A = A + lambda*sparse(i,j,feval(smoothFunc10,im,i,j,params)+feval(smoothFunc01,im,i,j,params),n,n);

i = 1:n;
for k=1:ns
    i(find(mod(i,h) == k)) = []; %up
end
j = i-ns;
[orientImg, distance] = buildOrientationImage(im,i(1),j(1),params);
params.orientImg = orientImg;
params.dist = distance;
T = T + sparse(i,ones(length(i),1), feval(smoothFunc10,im,i,j,params),n, 2);
T = T + sparse(j,ones(length(i),1)*2, feval(smoothFunc01,im,i,j,params),n, 2);
A = A + lambda*sparse(i,j,feval(smoothFunc10,im,i,j,params)+feval(smoothFunc01,im,i,j,params),n,n);

%%%% diagonals %%%
i = 1:n-ns*h; %right
for k=1:ns
    i(find(mod(i,h) == k)) = []; %up
end
j = i-ns+ns*h;
[orientImg, distance] = buildOrientationImage(im,i(1),j(1),params);
params.orientImg = orientImg;
params.dist = distance;
T = T + sparse(i,ones(length(i),1), feval(smoothFunc10,im,i,j,params),n, 2);
T = T + sparse(j,ones(length(i),1)*2, feval(smoothFunc01,im,i,j,params),n, 2);
A = A + lambda*sparse(i,j,feval(smoothFunc10,im,i,j,params)+feval(smoothFunc01,im,i,j,params),n,n);

i = 1:n-ns*h; %right
i(find(mod(i,h) == 0)) = []; %down
for k=1:ns-1
    i(find(mod(i,h) == h-k)) = []; %down
end
j = i+ns+ns*h;
[orientImg, distance] = buildOrientationImage(im,i(1),j(1),params);
params.orientImg = orientImg;
params.dist = distance;
T = T + sparse(i,ones(length(i),1), feval(smoothFunc10,im,i,j,params),n, 2);
T = T + sparse(j,ones(length(i),1)*2, feval(smoothFunc01,im,i,j,params),n, 2);
A = A + lambda*sparse(i,j,feval(smoothFunc10,im,i,j,params)+feval(smoothFunc01,im,i,j,params),n,n);

i = ns*h+ns:n; %left
i(find(mod(i,h) == 0)) = []; %down
for k=1:ns-1
    i(find(mod(i,h) == h-k)) = []; %down
end
j = i+ns-ns*h;
[orientImg, distance] = buildOrientationImage(im,i(1),j(1),params);
params.orientImg = orientImg;
params.dist = distance;
T = T + sparse(i,ones(length(i),1), feval(smoothFunc10,im,i,j,params),n, 2);
T = T + sparse(j,ones(length(i),1)*2, feval(smoothFunc01,im,i,j,params),n, 2);
A = A + lambda*sparse(i,j,feval(smoothFunc10,im,i,j,params)+feval(smoothFunc01,im,i,j,params),n,n);

i = ns*h+ns:n; %left
for k=1:ns
    i(find(mod(i,h) == k)) = []; %up
end
j = i-ns-ns*h;
[orientImg, distance] = buildOrientationImage(im,i(1),j(1),params);
params.orientImg = orientImg;
params.dist = distance;
T = T + sparse(i,ones(length(i),1), feval(smoothFunc10,im,i,j,params),n, 2);
T = T + sparse(j,ones(length(i),1)*2, feval(smoothFunc01,im,i,j,params),n, 2);
A = A + lambda*sparse(i,j,feval(smoothFunc10,im,i,j,params)+feval(smoothFunc01,im,i,j,params),n,n);


function [orientImg, distance] = buildOrientationImage(im,i,j,p)
[r1,c1] = ind2sub(size(im),i);
[r2,c2] = ind2sub(size(im),j);
b = [r1,c1] - [r2,c2];
distance = 1/sqrt(b(1).^2 + b(2).^2);

a = [0,1];
angle = -acos(a(1)*b(1)+a(2)*b(2)/(sqrt(a(1)^2+a(2)^2)*sqrt(b(1)^2+b(2)^2)));

if b(1) == 1
    angle = -angle;
end

d = zeros(p.cs);
d(1:end,round(p.cs/2)-round(p.ms/2):round(p.cs/2)+round(p.ms/2)) = 1;
%d = centeredRotate(d,angle);
d = imrotate(d,(angle/pi*180),'crop');
orientImg = normxcorr2_mex(double(d), 1-double(im), 'same');
