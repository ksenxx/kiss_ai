; Author: Koushik Sen (ksen@berkeley.edu)
; Contributors:
; Koushik Sen (ksen@berkeley.edu)
; add your name here
;
; The Windows setup program of KISS Sorcar, compiled by
; scripts/build-windows-exe.sh with NSIS (makensis), which passes
;   VERSION    the version of the checkout (2026.11.18)
;   PAYLOAD    the directory of the staged payload
;   RESOURCES  this directory
;   OUTPUT     the .exe to write
;
; It installs for the user who runs it and nobody else: it asks for no
; administrator rights, writes only into the user's profile and its
; registry entries are the user's own (HKCU).
;
; The payload (a copy of the sources, with install.sh) goes to
; %USERPROFILE%\.kiss\kiss_pkg, where the Mac package puts it too; then
; setup.ps1 runs its install.sh with Git Bash.  Run with /S it installs
; without showing a window and exits with a status other than 0 when
; the install failed.

Unicode true
ManifestDPIAware true
RequestExecutionLevel user
SetCompressor /SOLID lzma

!include "MUI2.nsh"
!include "LogicLib.nsh"

!define PRODUCT "KISS Sorcar"
!define UNINSTALL_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\KISS Sorcar"
; The port kiss-web listens on (KISS_WEB_PORT of install.sh).
!define DAEMON_PORT 8787

Name "${PRODUCT}"
OutFile "${OUTPUT}"
BrandingText "${PRODUCT} ${VERSION}"
ShowInstDetails show
ShowUninstDetails show

VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "${PRODUCT}"
VIAddVersionKey "FileDescription" "${PRODUCT} Setup"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "LegalCopyright" "Apache License 2.0"

!define MUI_ICON "${RESOURCES}/kiss-sorcar.ico"
!define MUI_UNICON "${RESOURCES}/kiss-sorcar.ico"
!define MUI_ABORTWARNING

!define MUI_WELCOMEPAGE_TITLE "Install ${PRODUCT} ${VERSION}"
!define MUI_WELCOMEPAGE_TEXT "This installs ${PRODUCT} for your Windows account only. It needs no administrator password.$\r$\n$\r$\nIt downloads Python, the Python packages and a browser for the agent (about 1 GB), so the PC must be online, and Git for Windows when the PC has none. This takes a few minutes.$\r$\n$\r$\nWhen it is done, ${PRODUCT} opens in a window of its own and is in the Start menu."
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_TITLE "${PRODUCT} is installed"
!define MUI_FINISHPAGE_TEXT "${PRODUCT} is in the Start menu and starts with Windows.$\r$\n$\r$\nSet your API keys in the app's Settings.$\r$\n$\r$\nThe log of this install is in $PROFILE\.kiss\install.log."
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

Function .onInit
    ; Always the same directory, whatever /D= says: the section below
    ; deletes it first.
    StrCpy $INSTDIR "$PROFILE\.kiss\kiss_pkg"
FunctionEnd

Section "Install"
    ; The payload of the version before: a file this version no longer
    ; has must not stay.
    RMDir /r "$INSTDIR"
    SetOutPath "$INSTDIR"
    File /r "${PAYLOAD}/*.*"

    ; The 64-bit PowerShell: this program is a 32-bit one, to which
    ; Windows shows its 64-bit system directory as "Sysnative".
    StrCpy $1 "$WINDIR\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
    ${IfNot} ${FileExists} "$1"
        StrCpy $1 "$SYSDIR\WindowsPowerShell\v1.0\powershell.exe"
    ${EndIf}
    DetailPrint "Installing ${PRODUCT}. This takes a few minutes."
    nsExec::ExecToLog '"$1" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "$INSTDIR\scripts\windows-exe\setup.ps1"'
    Pop $0
    ${If} $0 != "0"
        DetailPrint "${PRODUCT} could not be installed (status $0)."
        SetErrorLevel 1
        MessageBox MB_OK|MB_ICONSTOP "${PRODUCT} could not be installed.$\r$\n$\r$\nThe reason is in the window behind this message and in $PROFILE\.kiss\install.log." /SD IDOK
        Abort "${PRODUCT} could not be installed."
    ${EndIf}

    WriteUninstaller "$INSTDIR\uninstall.exe"
    WriteRegStr HKCU "${UNINSTALL_KEY}" "DisplayName" "${PRODUCT}"
    WriteRegStr HKCU "${UNINSTALL_KEY}" "DisplayVersion" "${VERSION}"
    WriteRegStr HKCU "${UNINSTALL_KEY}" "Publisher" "KISS Sorcar"
    WriteRegStr HKCU "${UNINSTALL_KEY}" "URLInfoAbout" "https://kisssorcar.github.io/"
    WriteRegStr HKCU "${UNINSTALL_KEY}" "DisplayIcon" "$INSTDIR\scripts\windows-exe\kiss-sorcar.ico"
    WriteRegStr HKCU "${UNINSTALL_KEY}" "InstallLocation" "$PROFILE\.kiss\kiss"
    WriteRegStr HKCU "${UNINSTALL_KEY}" "UninstallString" '"$INSTDIR\uninstall.exe"'
    WriteRegStr HKCU "${UNINSTALL_KEY}" "QuietUninstallString" '"$INSTDIR\uninstall.exe" /S'
    WriteRegDWORD HKCU "${UNINSTALL_KEY}" "NoModify" 1
    WriteRegDWORD HKCU "${UNINSTALL_KEY}" "NoRepair" 1
SectionEnd

Section "Uninstall"
    ; Not inside a directory that is deleted below.
    SetOutPath "$TEMP"
    ; The app stops its daemon and removes its start at logon and its
    ; Start menu shortcut (kiss/server/windows_install.py).
    StrCpy $1 "$PROFILE\.kiss\kiss\.venv\Scripts\python.exe"
    ${If} ${FileExists} "$1"
        nsExec::ExecToLog '"$1" -m kiss.server.windows_install uninstall --port ${DAEMON_PORT}'
        Pop $0
    ${EndIf}
    RMDir /r "$PROFILE\.kiss\kiss"
    RMDir /r "$PROFILE\.kiss\kiss_pkg"
    Delete "$PROFILE\.local\bin\sorcar"
    Delete "$PROFILE\.local\bin\sorcar.cmd"
    Delete "$PROFILE\.local\bin\rsorcar"
    Delete "$PROFILE\.local\bin\sorcar-docker"
    DeleteRegKey HKCU "${UNINSTALL_KEY}"
    DetailPrint "Your settings, API keys and task history stay in $PROFILE\.kiss."
SectionEnd
