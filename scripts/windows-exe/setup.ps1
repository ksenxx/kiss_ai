# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
#
# What the Windows setup program of KISS Sorcar runs once it has unpacked
# its payload (scripts/windows-exe/installer.nsi): the payload's
# install.sh, the one installer every way of installing KISS Sorcar ends
# in, with Git Bash.
#
# A PC without Git for Windows gets it first, into the user's profile
# (%LOCALAPPDATA%\Programs\Git, where Git's own installer puts a Git for
# one user): the portable edition, which needs no administrator rights.
# KISS Sorcar itself needs that Git afterwards, for git and for the bash
# its shell tool runs commands with.
#
# install.sh's output is passed on line by line with the characters
# outside ASCII removed: the setup program's window shows them as
# garbage.  The exit status is install.sh's, 1 when it could not be run.
#
#   -GitDir DIR   where to look for, and put, the Git of the profile
#                 (the tests name a directory of their own)

param(
    [string]$GitDir = (Join-Path $env:LOCALAPPDATA 'Programs\Git')
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is many times slower drawing one
$Payload = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$GitReleases = 'https://github.com/git-for-windows/git/releases'

function Find-GitBash {
    # The bash.exe of a Git for Windows: for all users, or in $GitDir.
    $roots = @($env:ProgramW6432, $env:ProgramFiles, ${env:ProgramFiles(x86)}) |
        Where-Object { $_ } | ForEach-Object { Join-Path $_ 'Git' }
    foreach ($root in @($roots) + $GitDir) {
        $bash = Join-Path $root 'bin\bash.exe'
        if (Test-Path $bash) { return $bash }
    }
    return $null
}

function Get-PortableGitUrl {
    # The download of the newest portable Git for this PC.  The page of
    # the latest release redirects to its tag, v2.52.0.windows.1 or
    # v2.52.0.windows.2, whose file is PortableGit-2.52.0-64-bit.7z.exe
    # or PortableGit-2.52.0.2-64-bit.7z.exe.
    $request = [Net.WebRequest]::Create("$GitReleases/latest")
    $request.AllowAutoRedirect = $false
    $response = $request.GetResponse()
    $location = $response.Headers['Location']
    $response.Close()
    if ($location -notmatch '/tag/(v(\d+\.\d+\.\d+)\.windows\.(\d+))$') {
        throw "no release tag in the answer of $GitReleases/latest ($location)"
    }
    $tag, $version, $build = $Matches[1], $Matches[2], $Matches[3]
    if ($build -ne '1') { $version = "$version.$build" }
    $arch = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'arm64' } else { '64-bit' }
    return "$GitReleases/download/$tag/PortableGit-$version-$arch.7z.exe"
}

function Install-PortableGit {
    # Download the portable Git and unpack it into $GitDir.
    $url = Get-PortableGitUrl
    $archive = Join-Path $env:TEMP 'kiss-portable-git.7z.exe'
    Write-Output "Downloading Git for Windows from $url"
    Invoke-WebRequest -Uri $url -OutFile $archive -UseBasicParsing
    Write-Output "Unpacking Git for Windows into $GitDir"
    $unpack = Start-Process -FilePath $archive -ArgumentList "-o`"$GitDir`"", '-y' -Wait -PassThru
    Remove-Item $archive -ErrorAction SilentlyContinue
    if ($unpack.ExitCode -ne 0) { throw "unpacking Git for Windows failed (status $($unpack.ExitCode))" }
}

function Get-WorkDir {
    # The directory the daemon serves: the one an earlier install
    # registered, else the user's profile.
    try {
        $service = Get-Content (Join-Path $env:USERPROFILE '.kiss\kiss-web-service.json') -Raw -ErrorAction Stop |
            ConvertFrom-Json
        if ($service.workdir -and (Test-Path $service.workdir -PathType Container)) { return $service.workdir }
    } catch { }
    return $env:USERPROFILE
}

function Invoke-Installer([string]$Bash) {
    # Run install.sh with $Bash; return its exit status.
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $Bash
    # Its errors with its output, in the order they are written.  The
    # script's path reaches bash in a variable: a profile directory may
    # hold a space or a quote.
    $info.Arguments = '-c "exec bash \"$KISS_SETUP_SCRIPT\" --non-interactive 2>&1"'
    $info.EnvironmentVariables['KISS_SETUP_SCRIPT'] = ($Payload -replace '\\', '/') + '/install.sh'
    $info.WorkingDirectory = Get-WorkDir
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardInput = $true
    $info.RedirectStandardOutput = $true
    $info.StandardOutputEncoding = [Text.Encoding]::UTF8
    # The app's window opens when the install is done; the Start menu
    # folder of its shortcut need not open as well.
    $info.EnvironmentVariables['KISS_NO_REVEAL'] = '1'
    $process = [Diagnostics.Process]::Start($info)
    $process.StandardInput.Close()
    while ($null -ne ($line = $process.StandardOutput.ReadLine())) {
        [Console]::Out.WriteLine(($line -replace '[^\x20-\x7E]', ''))
    }
    $process.WaitForExit()
    return $process.ExitCode
}

$bash = Find-GitBash
if (-not $bash) {
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Install-PortableGit
        $bash = Find-GitBash
        if (-not $bash) { throw "no bin\bash.exe in $GitDir after unpacking" }
    } catch {
        Write-Output "ERROR: Git for Windows could not be installed: $($_.Exception.Message)"
        Write-Output 'Install it from https://gitforwindows.org and run this setup again.'
        exit 1
    }
}
Write-Output "Running install.sh with $bash"
exit (Invoke-Installer $bash)
