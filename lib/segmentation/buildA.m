%this function builds connectivity matrix A and T for E(x,y) for GraphCut
%ASSUMPTIONS: E00 == E11 == 0 and E01 == E10
function [A,T] = buildA(im, lambda, ns, smoothFunc, params)
  n = numel(im);
  h = size(im,1);
  A = sparse(n,n);
  T = sparse(n,2);
  
  %#function E_smooth
  
  %links in im to neighbouring pixels
  i = [1:n-ns*h]; % right
  j = i+ns*h;
  T = T + sparse(i,ones(length(i),1), feval(smoothFunc,im,i,j,params), n, 2);
  T = T + sparse(j,ones(length(i),1)*2, feval(smoothFunc,im,i,j,params), n, 2);
  A = A+lambda*sparse(i,j,feval(smoothFunc,im,i,j,params),n,n); %A = A + lambda*At;
%  At = sparse(i+h,i,feval(smoothFunc,im(i),im(i+h),params,params),n,n); A = A + lambda*At;

i = [ns*h+1:n]; %left
j = i-ns*h;
  T = T + sparse(i,ones(length(i),1), feval(smoothFunc,im,i,j,params), n, 2);
  T = T + sparse(j,ones(length(i),1)*2, feval(smoothFunc,im,i,j,params), n, 2);
  A = A+lambda*sparse(i,j,feval(smoothFunc,im,i,j,params),n,n); %A = A + lambda*At;
%  At = sparse(i-h,i,feval(smoothFunc,im(i),im(i-h),params),n,n); A = A + lambda*At;

i = 1:n; 
i(find(mod(i,h) == 0)) = []; %down
  for k=1:ns-1
    i(find(mod(i,h) == h-k)) = []; %down
  end
  j = i+ns;
  T = T + sparse(i,ones(length(i),1), feval(smoothFunc,im,i,j,params), n, 2);
  T = T + sparse(j,ones(length(i),1)*2, feval(smoothFunc,im,i,j,params), n, 2);
  A = A+lambda*sparse(i,j,feval(smoothFunc,im,i,j,params),n,n); %A = A + lambda*At;
%  At = sparse(i+1,i,feval(smoothFunc,im(i),im(i+1),params),n,n); A = A + lambda*At;

i = 1:n; 
for k=1:ns
  i(find(mod(i,h) == k)) = []; %up
end
j = i-ns;
  T = T + sparse(i,ones(length(i),1), feval(smoothFunc,im,i,j,params), n, 2);
  T = T + sparse(j,ones(length(i),1)*2, feval(smoothFunc,im,i,j,params), n, 2);
  A = A+lambda*sparse(i,j,feval(smoothFunc,im,i,j,params),n,n); %A = A + lambda*At;
%  At = sparse(i-1,i,feval(smoothFunc,im(i),im(i-1),params),n,n); A = A + lambda*At;

i = 1:n-ns*h; 
for k=1:ns
  i(find(mod(i,h) == k)) = []; %up
end
j = i-ns+ns*h;
  T = T + sparse(i,ones(length(i),1), feval(smoothFunc,im,i,j,params), n, 2);
  T = T + sparse(j,ones(length(i),1)*2, feval(smoothFunc,im,i,j,params), n, 2);
  A = A+lambda*sparse(i,j,feval(smoothFunc,im,i,j,params),n,n); %A = A + lambda*At;
%  At = sparse(i-1+h,i,feval(smoothFunc,im(i),im(i-1+h),params),n,n); A = A + lambda*At;

i = 1:n-ns*h;
i(find(mod(i,h) == 0)) = []; %down
  for k=1:ns-1
    i(find(mod(i,h) == h-k)) = []; %down
  end
j = i+ns+ns*h;
T = T + sparse(i,ones(length(i),1), feval(smoothFunc,im,i,j,params), n, 2);
  T = T + sparse(j,ones(length(i),1)*2, feval(smoothFunc,im,i,j,params), n, 2);
  A = A+lambda*sparse(i,j,feval(smoothFunc,im,i,j,params),n,n); %A = A + lambda*At;
%  At = sparse(i+1+h,i,feval(smoothFunc,im(i),im(i+1+h),params),n,n); A = A + lambda*At;

i = ns*h+ns:n; 
i(find(mod(i,h) == 0)) = []; %down
for k=1:ns-1
  i(find(mod(i,h) == h-k)) = []; %down
end
j = i+ns-ns*h;
T = T + sparse(i,ones(length(i),1), feval(smoothFunc,im,i,j,params), n, 2);
T = T + sparse(j,ones(length(i),1)*2, feval(smoothFunc,im,i,j,params), n, 2);
A = A+lambda*sparse(i,j,feval(smoothFunc,im,i,j,params),n,n); %A = A + lambda*At;
%  At = sparse(i+1-h,i,feval(smoothFunc,im(i),im(i+1-h),params),n,n); A = A + lambda*At;

i = ns*h+ns:n; 
for k=1:ns
  i(find(mod(i,h) == k)) = []; %up
end
j = i-ns-ns*h;
T = T + sparse(i,ones(length(i),1), feval(smoothFunc,im,i,j,params), n, 2);
T = T + sparse(j,ones(length(i),1)*2, feval(smoothFunc,im,i,j,params), n, 2);
A = A+lambda*sparse(i,j,feval(smoothFunc,im,i,j,params),n,n); %A = A + lambda*At;
%  At = sparse(i-1-h,i,feval(smoothFunc,im(i),im(i-1-h),params),n,n); A = A + lambda*At;
  

  
