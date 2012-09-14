function sift_compile(type)
% SIFT_COMPILE  Compile MEX files
%   Compiling under Windows requires at least Visual C 6 or LCC. You
%   might try other compilers, but most likely you will need to edit
%   this file.

siftroot = fileparts(which('siftcompile')) ;
opts = { '-O', '-I.', '-g' } ;
opts = { opts{:}, '-v' } ;

if nargin < 1
    type = 'visualc' ;
end

switch computer
  case {'PCWIN', 'PCWIN64'}
    opts = {opts{:}, '-DWINDOWS'} ;

  case 'MAC'
    opts = {opts{:}, '-DMACOSX'} ;
    opts = {opts{:}, 'CFLAGS=\$CFLAGS -faltivec'} ;

  case {'MACI', 'MACI64'}
    opts = {opts{:}, '-DMACOSX'} ;

  case {'GLNX86', 'GLNXA64'}
    opts = {opts{:}, '-DLINUX' } ;

  otherwise
    error(['Unsupported architecture ', computer, '. Please edit this M-file to fix the issue.']) ;
end

mex('imsmooth.c',opts{:}) ;
mex('siftlocalmax.c',opts{:}) ;
mex('siftrefinemx.c',opts{:}) ;
mex('siftormx.c',opts{:}) ;
mex('siftdescriptor.c',opts{:}) ;
mex('siftmatch.c',opts{:}) ;




