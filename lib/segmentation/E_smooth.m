function vals = E_smooth(im,arg1, arg2, p)
  vals = 1/(p.sigma*sqrt(2*pi))*exp(-0.5*(im(arg1)-im(arg2)).^2 / p.sigma^2);