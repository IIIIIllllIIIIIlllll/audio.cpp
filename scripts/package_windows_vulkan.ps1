$ErrorActionPreference = "Stop"

$repoRoot = Split-Path $PSScriptRoot -Parent
$preset = "windows-vulkan-release"
$sourceBin = Join-Path (Join-Path $repoRoot "build") "$preset\bin"
$packageName = "audiocpp-windows-vulkan"
$outputDir = Join-Path (Join-Path $repoRoot "build") "prebuilt"
$stageDir = Join-Path $outputDir $packageName

if (-not (Test-Path (Join-Path $sourceBin "audiocpp_cli.exe")) -or
    -not (Test-Path (Join-Path $sourceBin "audiocpp_server.exe"))) {
    throw "Expected binaries were not found in $sourceBin"
}

if (Test-Path -LiteralPath $stageDir) { Remove-Item -LiteralPath $stageDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $stageDir | Out-Null
Copy-Item -Path (Join-Path $sourceBin "*") -Destination $stageDir -Recurse -Force

# Model specs and helper tools, mirroring package_windows_prebuilt.ps1.
$stageTools = Join-Path $stageDir "tools"
New-Item -ItemType Directory -Force -Path $stageTools | Out-Null
Copy-Item -LiteralPath (Join-Path $repoRoot "tools\model_manager_v2.py") -Destination $stageTools -Force
Copy-Item -LiteralPath (Join-Path $repoRoot "tools\model_manager_deprecated.py") -Destination $stageTools -Force
$communityTools = Join-Path $repoRoot "tools\community_models"
if (Test-Path -LiteralPath $communityTools) {
    Copy-Item -Path (Join-Path $communityTools "*") -Destination (Join-Path $stageTools "community_models") -Recurse -Force
}
Copy-Item -Path (Join-Path $repoRoot "model_specs\*") -Destination (Join-Path $stageDir "model_specs") -Recurse -Force
$modelManagerAssets = Join-Path $repoRoot "assets\model_manager"
if (Test-Path -LiteralPath $modelManagerAssets) {
    Copy-Item -Path (Join-Path $modelManagerAssets "*") -Destination (Join-Path $stageDir "assets\model_manager") -Recurse -Force
}

# VC++ runtime DLLs.
$vsRoot = "C:\Program Files\Microsoft Visual Studio\18\Community"
$crtDir = Get-ChildItem -Path (Join-Path $vsRoot "VC\Redist\MSVC\*\x64\Microsoft.VC143.CRT") -Directory |
    Sort-Object Name -Descending | Select-Object -First 1
$ompDir = Get-ChildItem -Path (Join-Path $vsRoot "VC\Redist\MSVC\*\x64\Microsoft.VC143.OpenMP") -Directory |
    Sort-Object Name -Descending | Select-Object -First 1
if ($null -eq $crtDir -or $null -eq $ompDir) { throw "VC++ redist folders were not found under $vsRoot" }
Copy-Item -LiteralPath (Join-Path $crtDir.FullName "MSVCP140.dll") -Destination $stageDir -Force
Copy-Item -LiteralPath (Join-Path $crtDir.FullName "VCRUNTIME140.dll") -Destination $stageDir -Force
Copy-Item -LiteralPath (Join-Path $crtDir.FullName "VCRUNTIME140_1.dll") -Destination $stageDir -Force
Copy-Item -LiteralPath (Join-Path $ompDir.FullName "VCOMP140.DLL") -Destination $stageDir -Force

$readme = @'
# audio.cpp Windows Vulkan build

Portable Windows build with the Vulkan backend enabled.

Contents:
- audiocpp_cli.exe: command line inference tool.
- audiocpp_server.exe: HTTP server with the embedded Web UI.
- model_specs/: model package specs used by the server Models page.
- tools/: legacy model manager helper scripts.
- MSVCP140.dll / VCRUNTIME140*.dll / VCOMP140.DLL: Visual C++ runtime.

Requirements:
- A GPU driver with Vulkan support (vulkan-1.dll comes with the driver).

Notes:
- This package targets GPU inference via Vulkan. Download models with the
  server Models page or place .gguf files next to the executables and pass
  them with --model.
'@
Set-Content -LiteralPath (Join-Path $stageDir "README.md") -Value $readme -Encoding UTF8

$zipPath = Join-Path $outputDir "$packageName.zip"
if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
Compress-Archive -Path (Join-Path $stageDir "*") -DestinationPath $zipPath -Force

$stageSize = (Get-ChildItem -LiteralPath $stageDir -File -Recurse | Measure-Object Length -Sum).Sum
$zipSize = (Get-Item -LiteralPath $zipPath).Length
[pscustomobject]@{
    Package = $packageName
    Directory = $stageDir
    Zip = $zipPath
    UncompressedMB = [math]::Round($stageSize / 1MB, 2)
    ZipMB = [math]::Round($zipSize / 1MB, 2)
} | Format-Table -AutoSize
