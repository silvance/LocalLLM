; Inno Setup script for LocalLLM. Wraps the PyInstaller-built single-file
; LocalLLM.exe into a proper Windows installer:
;   - Start Menu shortcut + (optional) Desktop shortcut
;   - Add/Remove Programs entry with version + uninstaller
;   - Per-user install (no admin elevation required) — examiners often
;     don't have local admin rights on the workstation
;   - First-run trigger of the smart-install configurator so DEFAULT_MODEL
;     reflects the actual hardware
;
; Build (on Windows, with Inno Setup 6 installed):
;     ISCC.exe installer\LocalLLM.iss
; Output:
;     installer\Output\LocalLLM-Setup-{version}.exe
;
; CI builds this automatically on every push to main; see
; .github/workflows/build.yml for the runner setup.

#ifndef MyAppVersion
  #define MyAppVersion "0.1.0"
#endif

#ifndef SourceExe
  #define SourceExe "..\dist\LocalLLM.exe"
#endif

[Setup]
AppId={{0E1E0F8A-1E5C-4F3D-8A5C-LOCALLLM00001}}
AppName=LocalLLM
AppVersion={#MyAppVersion}
AppPublisher=LocalLLM
AppSupportURL=https://github.com/silvance/LocalLLM
AppUpdatesURL=https://github.com/silvance/LocalLLM
DefaultDirName={userpf}\LocalLLM
DefaultGroupName=LocalLLM
DisableProgramGroupPage=yes
; Per-user install — no admin rights required.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputBaseFilename=LocalLLM-Setup-{#MyAppVersion}
OutputDir=Output
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Show the install size in the wizard so users know what they're committing to.
SetupLogging=yes
UninstallDisplayName=LocalLLM
UninstallDisplayIcon={app}\LocalLLM.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; \
    GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
; Single-file PyInstaller binary — that's the whole app + embedded Ollama.
; Models live next to the install dir (or under %LOCALAPPDATA%\LocalLLM\)
; — see paths.ollama_models_dir() resolution.
Source: "{#SourceExe}"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\LocalLLM";          Filename: "{app}\LocalLLM.exe"
Name: "{group}\LocalLLM Doctor";   Filename: "{app}\LocalLLM.exe"; Parameters: "doctor"; \
    Comment: "Diagnose installation issues"
Name: "{group}\Uninstall LocalLLM"; Filename: "{uninstallexe}"
Name: "{userdesktop}\LocalLLM";    Filename: "{app}\LocalLLM.exe"; Tasks: desktopicon

[Run]
; First-run: configure DEFAULT_MODEL based on detected hardware. Silent
; (--quiet) so the wizard doesn't pop a console; failures are non-fatal.
Filename: "{app}\LocalLLM.exe"; Parameters: "smart-install --quiet"; \
    StatusMsg: "Configuring for this machine's hardware..."; \
    Flags: runhidden waituntilterminated; \
    Check: ShouldRunSmartInstall

; Offer to launch the app after install completes.
Filename: "{app}\LocalLLM.exe"; Description: "Launch LocalLLM"; \
    Flags: nowait postinstall skipifsilent

[Code]
function ShouldRunSmartInstall(): Boolean;
begin
  // Always run on install. The script is idempotent — re-running it on
  // an upgrade just re-detects the hardware and rewrites the same .env.
  Result := True;
end;
