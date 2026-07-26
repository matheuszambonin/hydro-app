; installer.iss
; --------------------------------------------------------------------------
; Script do Inno Setup para o instalador do HydroPump (desktop, PySide6/Qt).
;
; Gera o fluxo padrão do Windows: Bem-vindo -> Avançar -> Local de instalação
; -> Avançar -> Atalhos -> Avançar -> Instalar -> Concluir. Nenhuma página
; customizada é adicionada, então o assistente usa exatamente as telas
; padrão do Inno Setup (WizardStyle=modern deixa com visual atual do Windows).
;
; Pré-requisito: a pasta dist\HydroPumpApp (gerada pelo PyInstaller a partir
; do app.spec) precisa existir antes de compilar este script.
;
; Compilar com:  ISCC.exe installer.iss
; --------------------------------------------------------------------------

#define MyAppName "HydroPump"
#define MyAppVersion "3.1.2"
#define MyAppPublisher "Substitua pelo seu nome ou empresa"
#define MyAppExeName "HydroPumpApp.exe"
#define MyAppSourceDir "dist\HydroPumpApp"

[Setup]
; Gere um GUID único para o seu app (PowerShell: [guid]::NewGuid()) e
; substitua abaixo. Mantenha o MESMO AppId em todas as versões futuras: é
; isso que permite que o Windows reconheça atualizações em vez de instalar
; duplicado.
AppId={{B6C1A9F4-8E2D-4C3B-9A1E-3F5D7E8C9A10}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
DisableWelcomePage=no
OutputDir=installer_output
OutputBaseFilename=Instalador_HydroPump
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
#ifexist "assets\app_icon.ico"
; Ícone opcional do instalador e do desinstalador — só incluído se existir.
SetupIconFile=assets\app_icon.ico
#endif
UninstallDisplayIcon={app}\{#MyAppExeName}
ArchitecturesInstallIn64BitMode=x64compatible
; "lowest" evita a tela de UAC, instalando apenas para o usuário atual —
; ideal para usuário leigo em máquina onde ele não é administrador. Troque
; para "admin" se precisar instalar para todos os usuários da máquina.
PrivilegesRequired=lowest

[Languages]
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: checkedonce

[Files]
; Copia TODA a pasta gerada pelo PyInstaller (exe + dlls + assets do
; streamlit + pacote hydropump/) preservando subpastas.
Source: "{#MyAppSourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Marca "Abrir HydroPump agora" já pré-selecionada na última tela (Concluir).
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Garante que arquivos gerados em tempo de execução dentro da pasta de
; instalação (se houver, ex.: hydropump.log) sejam removidos por completo
; ao desinstalar.
Type: filesandordirs; Name: "{app}"
