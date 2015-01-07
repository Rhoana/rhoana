# -*- mode: python -*-

block_cipher = None


a = Analysis(['viewer_3d.py'],
             pathex=['C:\\Users\\microway\\viewer\\rhoana\\Renderer'],
             hiddenimports=[],
             hookspath=None,
             runtime_hooks=None)
a.binaries = [x for x in a.binaries if not x[0].startswith("IPython")]
a.binaries = [x for x in a.binaries if not x[0].startswith("zmq")]
a.binaries = [x for x in a.binaries if not x[0].startswith("wx")]


pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='viewer_3d.exe',
          debug=False,
          strip=None,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=None,
               upx=True,
               name='viewer_3d')
