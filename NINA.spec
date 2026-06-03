# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ["qnegative/app.py"],
    pathex=[],
    binaries=[],
    datas=[
        ("LICENSE", "."),
        ("README.md", "."),
        ("logo/Banner.svg", "logo"),
        ("logo/NINA_LOGO.svg", "logo"),
        ("logo/NINA_TITLE.svg", "logo"),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "qnegative.tools",
        "sklearn",
        "scipy",
        "torch",
        "torchvision",
        "PIL",
        "imageio",
        "matplotlib",
        "skimage",
        "tkinter",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NINA",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="NINA",
)
