param(
    [string]$VenvPython = ".venv\\Scripts\\python.exe",
    [string]$SpecPath = "packaging\\pyinstaller\\aura.spec",
    [string]$RuntimeRoot = ".runtime",
    [string]$ReleaseName = "aura-release",
    [string]$PyInstallerVersion = "6.14.2",
    [switch]$IncludeNvidia,
    [switch]$CreateZip,
    [switch]$CreateNvidiaOverlay,
    [switch]$SkipBuild,
    [switch]$SkipAssemble
)

$ErrorActionPreference = "Stop"

function Assert-PathExists {
    param([string]$PathValue, [string]$Label)
    if (-not (Test-Path $PathValue)) {
        throw "$Label not found: $PathValue"
    }
}

function Invoke-RobocopySafe {
    param(
        [string]$Source,
        [string]$Destination,
        [string[]]$ExtraArgs = @()
    )

    Assert-PathExists -PathValue $Source -Label "Robocopy source"
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null

    $args = @(
        $Source,
        $Destination,
        "/E",
        "/NFL",
        "/NDL",
        "/NJH",
        "/NJS",
        "/NP"
    ) + $ExtraArgs

    & robocopy @args | Out-Null
    $code = $LASTEXITCODE
    if ($code -ge 8) {
        throw "Robocopy failed with exit code $code while copying '$Source' -> '$Destination'."
    }
}

function Test-PythonModuleAvailable {
    param(
        [string]$PythonPath,
        [string]$ModuleName
    )

    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $PythonPath -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$ModuleName') else 1)" *> $null
        return $LASTEXITCODE -eq 0
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
}

function Assert-PythonModulesAbsent {
    param(
        [string]$PythonPath,
        [string[]]$ModuleNames
    )

    $presentModules = @()
    foreach ($moduleName in $ModuleNames) {
        if (Test-PythonModuleAvailable -PythonPath $PythonPath -ModuleName $moduleName) {
            $presentModules += $moduleName
        }
    }

    if ($presentModules.Count -gt 0) {
        throw (
            "Release venv is not clean. These build-only or excluded runtime modules are importable: " +
            ($presentModules -join ", ") +
            ". Use a clean venv such as .venv-release-gpu-onnx with runtime.txt and optional-vision-onnx-cuda.txt only."
        )
    }
}

function Assert-OnnxRuntimeGpuEnvironment {
    param([string]$PythonPath)

    $script = @'
from importlib import metadata
import sys

try:
    metadata.version("onnxruntime-gpu")
except metadata.PackageNotFoundError:
    raise SystemExit("onnxruntime-gpu is required for the GPU ONNX release venv.")

try:
    metadata.version("onnxruntime")
except metadata.PackageNotFoundError:
    pass
else:
    raise SystemExit("Do not install both onnxruntime and onnxruntime-gpu in the release venv.")

import onnxruntime as ort

providers = list(ort.get_available_providers())
if "CPUExecutionProvider" not in providers:
    raise SystemExit(f"ONNX Runtime CPUExecutionProvider is missing: {providers!r}")

print("ONNX Runtime GPU release preflight OK: providers=" + ",".join(providers))
'@

    $output = $script | & $PythonPath - 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "ONNX Runtime GPU preflight failed: $output"
    }
    if ($output) {
        Write-Host $output
    }
}

function Assert-OcrModelBundle {
    param([string]$ModelsRoot)

    $bundleDir = Join-Path $ModelsRoot "ppocrv5_server"
    Assert-PathExists -PathValue $bundleDir -Label "OCR ONNX model bundle"

    foreach ($fileName in @("ocr.meta.json", "det.onnx", "rec.onnx", "doc_orientation.onnx", "textline_orientation.onnx")) {
        Assert-PathExists -PathValue (Join-Path $bundleDir $fileName) -Label "OCR ONNX model file"
    }
}

function Get-FileProductVersion {
    param([string]$PathValue)

    if (-not (Test-Path $PathValue)) {
        return $null
    }

    $item = Get-Item -LiteralPath $PathValue
    $rawVersion = $item.VersionInfo.ProductVersion
    if (-not $rawVersion) {
        $rawVersion = $item.VersionInfo.FileVersion
    }
    if (-not $rawVersion) {
        return $null
    }

    $match = [regex]::Match($rawVersion, "\d+(\.\d+){1,3}")
    if (-not $match.Success) {
        return $null
    }

    try {
        return [version]$match.Value
    }
    catch {
        return $null
    }
}

function Update-MsvcRuntimeForOnnxRuntime {
    param([string]$RuntimeDir)

    $targetPath = Join-Path $RuntimeDir "_internal\msvcp140.dll"
    if (-not (Test-Path $targetPath)) {
        return
    }

    $sourcePath = Join-Path $env:SystemRoot "System32\msvcp140.dll"
    if (-not (Test-Path $sourcePath)) {
        Write-Warning "System MSVC runtime was not found at $sourcePath; packaged onnxruntime-gpu may require a newer VC runtime on target machines."
        return
    }

    $targetVersion = Get-FileProductVersion -PathValue $targetPath
    $sourceVersion = Get-FileProductVersion -PathValue $sourcePath
    if ($null -eq $sourceVersion) {
        Write-Warning "Could not determine System32 msvcp140.dll version; leaving packaged MSVC runtime unchanged."
        return
    }

    if ($null -eq $targetVersion -or $sourceVersion -gt $targetVersion) {
        Write-Host "Updating packaged msvcp140.dll for ONNX Runtime: $targetVersion -> $sourceVersion"
        Copy-Item -LiteralPath $sourcePath -Destination $targetPath -Force
    }
}

function Copy-OcrModels {
    param(
        [string]$Source,
        [string]$Destination
    )

    Assert-OcrModelBundle -ModelsRoot $Source
    Invoke-RobocopySafe `
        -Source $Source `
        -Destination $Destination `
        -ExtraArgs @("/XD", "__pycache__", ".pytest_cache", "/XF", "*.pyc", "*.pyo")
    Assert-OcrModelBundle -ModelsRoot $Destination
}

function Copy-PlanPackages {
    param(
        [string]$Source,
        [string]$Destination
    )

    Assert-PathExists -PathValue $Source -Label "Plans directory"
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null

    Get-ChildItem -LiteralPath $Source -File -Filter "*.py" |
        ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $Destination $_.Name) -Force
        }

    $excludedPlanDirectoryNames = @("logs", "__pycache__", ".pytest_cache")
    $planPackages = Get-ChildItem -LiteralPath $Source -Directory |
        Where-Object {
            $excludedPlanDirectoryNames -notcontains $_.Name -and
            (Test-Path (Join-Path $_.FullName "manifest.yaml"))
        }

    foreach ($planPackage in $planPackages) {
        Invoke-RobocopySafe `
            -Source $planPackage.FullName `
            -Destination (Join-Path $Destination $planPackage.Name) `
            -ExtraArgs @("/XD", "__pycache__", ".pytest_cache", "ocr_model", "/XF", "*.pyc", "*.pyo")
    }
}

function Ensure-PyInstaller {
    param(
        [string]$PythonPath,
        [string]$Version
    )

    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $PythonPath -c "import PyInstaller" *> $null
        $hasPyInstaller = $LASTEXITCODE -eq 0
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if ($hasPyInstaller) {
        return
    }

    Write-Host "Installing PyInstaller $Version into build venv ..."
    & $PythonPath -m pip install "pyinstaller==$Version"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install PyInstaller $Version."
    }
}

function Get-NvidiaPackageRoot {
    param([string]$PythonPath)

    $script = @'
from pathlib import Path
import site

for site_root in site.getsitepackages():
    candidate = Path(site_root) / "nvidia"
    if candidate.is_dir():
        print(candidate)
        break
'@
    $path = ($script | & $PythonPath - 2>$null | Select-Object -First 1)
    if (-not $path) {
        throw "NVIDIA Python runtime packages were not found in the release venv. Install the required nvidia-* CUDA/cuDNN wheels before using -CreateNvidiaOverlay."
    }
    return [string]$path
}

function Assert-NvidiaRuntimeOverlayBundle {
    param([string]$NvidiaRoot)

    Assert-PathExists -PathValue $NvidiaRoot -Label "NVIDIA Python runtime root"
    $requiredDlls = @(
        "cublas\bin\cublas64_12.dll",
        "cublas\bin\cublasLt64_12.dll",
        "cuda_runtime\bin\cudart64_12.dll",
        "cudnn\bin\cudnn64_9.dll",
        "cufft\bin\cufft64_11.dll"
    )

    $missingDlls = @()
    foreach ($relativePath in $requiredDlls) {
        $candidate = Join-Path $NvidiaRoot $relativePath
        if (-not (Test-Path $candidate)) {
            $missingDlls += $relativePath
        }
    }

    if ($missingDlls.Count -gt 0) {
        throw (
            "NVIDIA runtime overlay is incomplete. Missing: " +
            ($missingDlls -join ", ") +
            ". Install requirements\\optional-nvidia-runtime-cu12.txt into the release venv before using -CreateNvidiaOverlay."
        )
    }
}

function New-ZipArchive {
    param(
        [string]$SourcePath,
        [string]$DestinationPath
    )

    Assert-PathExists -PathValue $SourcePath -Label "Zip source"
    if (Test-Path $DestinationPath) {
        Remove-Item -LiteralPath $DestinationPath -Force
    }

    Write-Host "Creating zip archive: $DestinationPath"
    Compress-Archive -Path $SourcePath -DestinationPath $DestinationPath -Force
}

function New-NvidiaRuntimeOverlay {
    param(
        [string]$PythonPath,
        [string]$RuntimeRootPath,
        [string]$ReleaseName
    )

    $nvidiaSource = Get-NvidiaPackageRoot -PythonPath $PythonPath
    Assert-NvidiaRuntimeOverlayBundle -NvidiaRoot $nvidiaSource

    $overlayRoot = Join-Path $RuntimeRootPath "release\\$ReleaseName-nvidia-overlay"
    $overlayRuntimeInternal = Join-Path $overlayRoot "runtime\\_internal"
    $overlayNvidiaDir = Join-Path $overlayRuntimeInternal "nvidia"
    $overlayZip = Join-Path $RuntimeRootPath "release\\$ReleaseName-nvidia-overlay.zip"

    if (Test-Path $overlayRoot) {
        Remove-Item -LiteralPath $overlayRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $overlayRuntimeInternal | Out-Null

    Write-Host "Assembling NVIDIA runtime overlay ..."
    Invoke-RobocopySafe `
        -Source $nvidiaSource `
        -Destination $overlayNvidiaDir `
        -ExtraArgs @("/XD", "__pycache__", ".pytest_cache", "/XF", "*.pyc", "*.pyo")

    @(
        "NVIDIA runtime overlay for $ReleaseName"
        ""
        "Extract this archive into the release root so it creates:"
        "runtime\\_internal\\nvidia"
        ""
        "The main package intentionally keeps these CUDA/cuDNN runtime libraries external."
    ) | Set-Content -Path (Join-Path $overlayRoot "NVIDIA-RUNTIME-OVERLAY.txt") -Encoding UTF8

    if (Test-Path $overlayZip) {
        Remove-Item -LiteralPath $overlayZip -Force
    }
    Write-Host "Creating NVIDIA runtime overlay zip: $overlayZip"
    Compress-Archive -Path (Join-Path $overlayRoot "*") -DestinationPath $overlayZip -Force
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPythonPath = Join-Path $RepoRoot $VenvPython
$SpecFilePath = Join-Path $RepoRoot $SpecPath
$RuntimeRootPath = Join-Path $RepoRoot $RuntimeRoot
$DistPath = Join-Path $RuntimeRootPath "dist"
$WorkPath = Join-Path $RuntimeRootPath "build\\pyinstaller"
$ReleaseRoot = Join-Path $RuntimeRootPath "release\\$ReleaseName"
$BuiltRuntimeDir = Join-Path $DistPath "aura"
$ReleaseRuntimeDir = Join-Path $ReleaseRoot "runtime"
$ReleasePlansDir = Join-Path $ReleaseRoot "plans"
$ReleaseOcrModelsDir = Join-Path $ReleaseRoot "models\\ocr"
$RunTemplate = Join-Path $RepoRoot "packaging\\templates\\run.ps1"
$ConfigTemplate = Join-Path $RepoRoot "packaging\\templates\\config.yaml"
$SourcePlansDir = Join-Path $RepoRoot "plans"
$SourceOcrModelsDir = Join-Path $RepoRoot "models\\ocr"
$SourceLicense = Join-Path $RepoRoot "LICENSE"
$SourceReadme = Join-Path $RepoRoot "README.md"

Assert-PathExists -PathValue $VenvPythonPath -Label "Venv python"
Assert-PathExists -PathValue $SpecFilePath -Label "PyInstaller spec"
Assert-PathExists -PathValue $RunTemplate -Label "Run script template"
Assert-PathExists -PathValue $ConfigTemplate -Label "Config template"
Assert-PathExists -PathValue $SourcePlansDir -Label "Plans directory"
Assert-OcrModelBundle -ModelsRoot $SourceOcrModelsDir

$env:PYTHONNOUSERSITE = "1"
$env:AURA_PKG_INCLUDE_NVIDIA = if ($IncludeNvidia) { "1" } else { "0" }

Ensure-PyInstaller -PythonPath $VenvPythonPath -Version $PyInstallerVersion

if (-not $SkipBuild) {
    Assert-PythonModulesAbsent -PythonPath $VenvPythonPath -ModuleNames @(
        "paddle",
        "paddleocr",
        "paddlex",
        "torch",
        "torchvision",
        "ultralytics",
        "PySide6",
        "shiboken6"
    )
    Assert-OnnxRuntimeGpuEnvironment -PythonPath $VenvPythonPath
}

if (-not $SkipBuild) {
    Write-Host "Building Aura runtime with PyInstaller ..."
    New-Item -ItemType Directory -Force -Path $DistPath | Out-Null
    New-Item -ItemType Directory -Force -Path $WorkPath | Out-Null

    & $VenvPythonPath -m PyInstaller `
        --noconfirm `
        --clean `
        --distpath $DistPath `
        --workpath $WorkPath `
        $SpecFilePath

    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed with exit code $LASTEXITCODE."
    }
}

if (-not $SkipAssemble) {
    Assert-PathExists -PathValue $BuiltRuntimeDir -Label "Built runtime directory"
    Update-MsvcRuntimeForOnnxRuntime -RuntimeDir $BuiltRuntimeDir

    if (Test-Path $ReleaseRoot) {
        Remove-Item -LiteralPath $ReleaseRoot -Recurse -Force
    }

    New-Item -ItemType Directory -Force -Path $ReleaseRoot | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $ReleaseRoot "logs") | Out-Null

    Write-Host "Assembling release root ..."
    Invoke-RobocopySafe -Source $BuiltRuntimeDir -Destination $ReleaseRuntimeDir
    Update-MsvcRuntimeForOnnxRuntime -RuntimeDir $ReleaseRuntimeDir
    Copy-PlanPackages -Source $SourcePlansDir -Destination $ReleasePlansDir
    Copy-OcrModels -Source $SourceOcrModelsDir -Destination $ReleaseOcrModelsDir

    Copy-Item -LiteralPath $RunTemplate -Destination (Join-Path $ReleaseRoot "run.ps1") -Force

    $releaseConfigPath = Join-Path $ReleaseRoot "config.yaml"
    if (-not (Test-Path $releaseConfigPath)) {
        Copy-Item -LiteralPath $ConfigTemplate -Destination $releaseConfigPath -Force
    }

    if (Test-Path $SourceLicense) {
        Copy-Item -LiteralPath $SourceLicense -Destination (Join-Path $ReleaseRoot "LICENSE") -Force
    }
    if (Test-Path $SourceReadme) {
        Copy-Item -LiteralPath $SourceReadme -Destination (Join-Path $ReleaseRoot "README.md") -Force
    }

    $nvidiaRuntimeMode = if ($IncludeNvidia) {
        "bundled"
    } elseif ($CreateNvidiaOverlay) {
        "overlay"
    } else {
        "external"
    }

    $summaryPath = Join-Path $ReleaseRoot "BUILD-INFO.txt"
    @(
        "release_name=$ReleaseName"
        "built_at_utc=$([DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ'))"
        "include_nvidia=$($IncludeNvidia.IsPresent)"
        "create_zip=$($CreateZip.IsPresent)"
        "create_nvidia_overlay=$($CreateNvidiaOverlay.IsPresent)"
        "ocr_backend=onnxruntime"
        "gpu_backend=onnxruntime-gpu"
        "paddle_stack=false"
        "nvidia_runtime=$nvidiaRuntimeMode"
        "base_path_mode=release_root"
        "entrypoint=run.ps1"
    ) | Set-Content -Path $summaryPath -Encoding UTF8

    Write-Host "Release assembled at: $ReleaseRoot"
}

if ($CreateZip) {
    Update-MsvcRuntimeForOnnxRuntime -RuntimeDir $ReleaseRuntimeDir
    New-ZipArchive `
        -SourcePath $ReleaseRoot `
        -DestinationPath (Join-Path $RuntimeRootPath "release\\$ReleaseName.zip")
}

if ($CreateNvidiaOverlay) {
    New-NvidiaRuntimeOverlay `
        -PythonPath $VenvPythonPath `
        -RuntimeRootPath $RuntimeRootPath `
        -ReleaseName $ReleaseName
}
