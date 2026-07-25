# -*- mode: python ; coding: utf-8 -*-
"""
app.spec
--------
Especificação do PyInstaller para o HydroPump.

Gera uma build em PASTA (onedir, via COLLECT) em vez de um único .exe (onefile).
Isso é proposital: apps Streamlit têm muitos arquivos estáticos (frontend em
JS/CSS) e dependências pesadas (pandas, numpy, scipy, matplotlib, pyarrow).
Em modo onefile o Windows precisa descompactar tudo em uma pasta temporária
TODA VEZ que o programa abre, deixando o início lento. Em modo onedir a
aplicação abre quase instantaneamente. O Inno Setup (installer.iss) empacota
essa pasta inteira dentro do instalador.

Rodar com:  pyinstaller app.spec --noconfirm --clean
"""

from PyInstaller.utils.hooks import collect_all

block_cipher = None

datas = []
binaries = []
hiddenimports = []

# Pacotes que precisam de seus arquivos de dados internos (templates, .json,
# static/, metadados) além do código Python puro. streamlit e pyarrow são os
# mais sensíveis a isso; os demais estão aqui por segurança.
PACKAGES_COLLECT_ALL = [
    "streamlit",
    "altair",       # dependência interna do streamlit para alguns componentes
    "pyarrow",      # streamlit usa para serializar DataFrames
    "matplotlib",
    "seaborn",
    "pandas",
    "numpy",
    "scipy",
]

for pkg in PACKAGES_COLLECT_ALL:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Imports que o PyInstaller às vezes não detecta sozinho por serem
# carregados dinamicamente dentro do próprio Streamlit.
hiddenimports += [
    "streamlit.web.bootstrap",
    "streamlit.runtime.scriptrunner.magic_funcs",
]

# --------------------------------------------------------------------
# Arquivos do PRÓPRIO projeto (ajuste os caminhos conforme o seu repo).
# Ficam disponíveis em tempo de execução em sys._MEIPASS/app/...
# (ver resource_path() em run_app.py).
# --------------------------------------------------------------------
datas += [
    ("app.py", "app"),        # ponto de entrada do Streamlit
    ("src", "app/src"),       # módulos: plotting.py, hydro_math.py, schematics.py
]

# Opcional: se existir uma pasta .streamlit/config.toml com tema/cores
# customizadas, inclua-a também. Comente a linha se não existir no repo.
# datas += [(".streamlit", "app/.streamlit")]

a = Analysis(
    ["run_app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    upx=True,
    console=False,  # janela "windowed": nenhum terminal aparece para o usuário
    icon="assets/app_icon.ico",  # troque pelo seu ícone; remova a linha se não tiver um
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="HydroPumpApp",  # resultado final em dist/HydroPumpApp/
)
