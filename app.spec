# -*- mode: python ; coding: utf-8 -*-
"""
app.spec
--------
Especificação do PyInstaller para o HydroPump (desktop, PySide6/Qt).

Diferente da versão anterior (Streamlit), não há servidor local nem
arquivos estáticos de frontend a empacotar — o PyInstaller já tem hooks
maduros para PySide6, pandas, numpy, scipy e matplotlib, então a receita
aqui é enxuta por design.

Build em PASTA (onedir, via COLLECT): abre quase instantaneamente e é o
que o Inno Setup (installer.iss) empacota.

Rodar com:  pyinstaller app.spec --noconfirm --clean
"""

import os

from PyInstaller.utils.hooks import copy_metadata

block_cipher = None

datas = []
binaries = []
hiddenimports = []

# numpy/scipy/pandas/matplotlib/PySide6 já têm hooks nativos maduros no
# PyInstaller; só precisamos garantir os metadados de versão
# (importlib.metadata), que alguma dessas libs consulta em runtime.
for pkg in ("numpy", "scipy", "pandas", "matplotlib", "seaborn"):
    datas += copy_metadata(pkg)

_icon_path = "assets/app_icon.ico"
_icon = _icon_path if os.path.exists(_icon_path) else None

a = Analysis(
    ["main.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Bindings Qt concorrentes -- evita ambiguidade de qual o
        # Matplotlib deve carregar (ver QT_API em ui_qt/widgets/mpl_canvas.py).
        "PyQt5", "PyQt6", "PySide2",
        # Modulos Qt pesados que este app nao usa; excluir reduz o
        # instalador final em dezenas de MB.
        "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineCore",
        "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuickWidgets",
        "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets",
        "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtSensors",
        "PySide6.QtSerialPort", "PySide6.QtPositioning", "PySide6.QtLocation",
        "PySide6.QtDesigner", "PySide6.QtPdf", "PySide6.QtPdfWidgets",
        "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.Qt3DInput",
        # Nao usados neste projeto.
        "tkinter", "IPython", "notebook", "jupyter",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HydroPumpApp",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=_icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="HydroPumpApp",
)
