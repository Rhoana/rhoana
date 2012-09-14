function sift_compile(type)
% SIFT_COMPILE  Compile MEX files
%   Compiling under Windows requires at least Visual C 6 or LCC. You
%   might try other compilers, but most likely you will need to edit
%   this file.

siftroot = fileparts(which('siftcompile')) ;
opts = { '-O', '-I.' } ;
%opts = { opts{:}, '-v' } ;

if nargin < 1
    type = 'visualc' ;
end

switch computer
  case 'PCWIN64'
    warning('NOTE: compiling has been tested only with Visual C 6-7 and LCC') ;
%    switch type
%      case 'visualc'
%        lib{1}=[matlabroot '\extern\lib\win32\microsoft\libmwlapack.lib'] ;
%        lib{2}=[matlabroot '\extern\lib\win32\microsoft\msvc60\libmwlapack.lib'];
%        lib{3}=[matlabroot '\extern\lib\win32\microsoft\msvc71\libmwlapack.lib'];
%      case 'lcc'
        lib{1}=[matlabroot '\extern\lib\win64\microsoft'] ;
%    end
    found=0;
    for k=1:length(lib)
      fprintf('Trying LAPACK lib ''%s''\n',lib{k}) ;
      found=exist(lib{k}) ;
      if found ~= 0
        break ;
      end
    end
    if found == 0
      error('Could not find LAPACK library. Please edit this M-file to fix the issue.');
    end
    opts = {opts{:}, '-DWINDOWS'} ;
    opts = {opts{:}, lib{k}} ;
    
  case 'MAC'
    opts = {opts{:}, '-DMACOSX'} ;
    opts = {opts{:}, 'CFLAGS=\$CFLAGS -faltivec'} ;
  
  case 'GLNX86'
    fprintf('This is a GLNX86 architecture');
    opts = {opts{:}, '/usr/share/matlab/bin/glnx86/libmwlapack.so'};
    opts = {opts{:}, '-DLINUX' } ;
        
  otherwise
    error(['Unsupported architecture ', computer, '. Please edit this M-mfile to fix the issue.']) ;    
end

mex('imsmooth.c',opts{:}) ;
mex('siftlocalmax.c',opts{:}) ;
mex('siftrefinemx.c',opts{:}) ;
mex('siftormx.c',opts{:}) ;
mex('siftdescriptor.c',opts{:}) ;
mex('siftmatch.c',opts{:}) ;

    


