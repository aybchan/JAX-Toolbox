# Build the pinned upstream NCCL shared library with ARM64 MSVC and native NVCC.
# The generated SDK is consumed by the Windows CUDA Bazel repositories.
param(
    [Parameter(Mandatory = $true)][string]$SourceRoot,
    [string]$BuildRoot = 'C:/b/nccl',
    [string]$CudaPath = 'C:/cuda',
    [string]$InstallPrefix = 'C:/nccl',
    [ValidateRange(1, 64)][int]$Jobs = 4,
    [string]$VcDirectory = $env:BAZEL_VC
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$expectedCommit = 'ad2d192c1f279aefcaf340f67e07880b65a1116b'
$expectedVersion = '2.32.6'
$architectures = '75-real;80-real;90-real;100-real;110-real;120'

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program failed with exit code $LASTEXITCODE"
    }
}

function Require-File {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required NCCL build input or output is missing: $Path"
    }
}

function Assert-Arm64Pe {
    param([string]$Path)
    Require-File $Path
    $stream = [IO.File]::OpenRead($Path)
    $reader = [IO.BinaryReader]::new($stream)
    try {
        if ($stream.Length -lt 64 -or $reader.ReadUInt16() -ne 0x5a4d) {
            throw "Not a Windows PE binary: $Path"
        }
        $stream.Position = 0x3c
        $offset = $reader.ReadUInt32()
        if ($offset -lt 64 -or $offset -gt $stream.Length - 6) {
            throw "Invalid PE header offset: $Path"
        }
        $stream.Position = $offset
        if ($reader.ReadUInt32() -ne 0x00004550 -or $reader.ReadUInt16() -ne 0xaa64) {
            throw "Expected a native Windows ARM64 PE binary: $Path"
        }
    } finally {
        $reader.Dispose()
        $stream.Dispose()
    }
}

if (-not $IsWindows) { throw 'The NCCL build requires Windows ARM64' }
if ([Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne 'Arm64') {
    throw 'The NCCL build requires a native Windows ARM64 runner'
}
$SourceRoot = (Resolve-Path -LiteralPath $SourceRoot).Path
$CudaPath = (Resolve-Path -LiteralPath $CudaPath).Path
$VcDirectory = (Resolve-Path -LiteralPath $VcDirectory).Path
$actualCommit = (& git -C $SourceRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $expectedCommit) {
    throw "Expected NCCL commit $expectedCommit; found $actualCommit"
}
$versionFile = Join-Path $SourceRoot 'makefiles/version.mk'
Require-File $versionFile
$versionText = [IO.File]::ReadAllText($versionFile)
$parts = foreach ($part in @('MAJOR', 'MINOR', 'PATCH')) {
    $match = [regex]::Match($versionText, "(?m)^NCCL_$part\s*:=\s*(\d+)\s*$")
    if (-not $match.Success) { throw "Cannot read NCCL_$part from $versionFile" }
    $match.Groups[1].Value
}
if (($parts -join '.') -ne $expectedVersion) {
    throw "Expected NCCL $expectedVersion; found $($parts -join '.')"
}
Require-File (Join-Path $SourceRoot 'CMakeLists.txt')
Require-File (Join-Path $SourceRoot 'LICENSE.txt')

New-Item -ItemType Directory -Force $BuildRoot | Out-Null
$BuildRoot = (Resolve-Path -LiteralPath $BuildRoot).Path
$InstallPrefix = [IO.Path]::GetFullPath($InstallPrefix)
$vcvars = Join-Path $VcDirectory 'Auxiliary/Build/vcvarsall.bat'
Require-File $vcvars

# Import the same ARM64 MSVC environment used by the working JNI build.
# Keep it local to this workflow step; the caller exports the completed SDK.
$environmentBatch = Join-Path $BuildRoot 'msvc-environment.cmd'
@(
    '@echo off',
    "call `"$vcvars`" arm64 >nul",
    'if errorlevel 1 exit /b %errorlevel%',
    'set'
) | Set-Content -Encoding ascii $environmentBatch
$environmentLines = & cmd.exe /d /c $environmentBatch
if ($LASTEXITCODE -ne 0) { throw 'Could not initialize the ARM64 MSVC environment' }
foreach ($line in $environmentLines) {
    if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}
$compiler = (Get-Command cl.exe -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
$nvcc = Join-Path $CudaPath 'bin/nvcc.exe'
Assert-Arm64Pe $nvcc
foreach ($relative in @('include/cuda_runtime.h', 'include/cccl/cuda/std/version',
                        'lib/arm64/cudart.lib', 'lib/arm64/cudart_static.lib',
                        'lib/arm64/cudadevrt.lib')) {
    Require-File (Join-Path $CudaPath $relative)
}
$env:CUDA_PATH = $CudaPath
$env:PATH = "$CudaPath/bin;$CudaPath/bin/arm64;$env:PATH"

# Confirm the selected cl.exe emits ARM64 objects before configuring CUDA.
$probeSource = Join-Path $BuildRoot 'arm64-compiler-probe.c'
$probeObject = Join-Path $BuildRoot 'arm64-compiler-probe.obj'
@'
#ifndef _M_ARM64
#error The NCCL host compiler must target Windows ARM64.
#endif
int nccl_arm64_compiler_probe(void) { return _MSC_VER; }
'@ | Set-Content -Encoding ascii $probeSource
Invoke-Checked $compiler @('/nologo', '/c', $probeSource, "/Fo$probeObject")
$probeBytes = [IO.File]::ReadAllBytes($probeObject)
if ($probeBytes.Length -lt 20 -or [BitConverter]::ToUInt16($probeBytes, 0) -ne 0xaa64) {
    throw 'The selected MSVC compiler did not emit an ARM64 COFF object'
}

# The runner or Visual Studio provides CMake/Ninja; no CUDA VS integration is needed.
$visualStudio = Split-Path $VcDirectory
$cmakeCommand = Get-Command cmake.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
$ninjaCommand = Get-Command ninja.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
$cmake = if ($cmakeCommand) { $cmakeCommand.Source } else {
    Join-Path $visualStudio 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
}
$ninja = if ($ninjaCommand) { $ninjaCommand.Source } else {
    Join-Path $visualStudio 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe'
}
Require-File $cmake
Require-File $ninja
Invoke-Checked $cmake @('--version')
Invoke-Checked $ninja @('--version')
Invoke-Checked $nvcc @('--version')
$python = (Get-Command python.exe -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source

# Use forward slashes for CMake cache values, including paths with spaces.
$sourceCmake = $SourceRoot.Replace('\', '/')
$buildCmake = $BuildRoot.Replace('\', '/')
$cudaCmake = $CudaPath.Replace('\', '/')
$compilerCmake = $compiler.Replace('\', '/')
$configure = @(
    '-S', $sourceCmake, '-B', $buildCmake, '-G', 'Ninja',
    "-DCMAKE_MAKE_PROGRAM=$($ninja.Replace('\', '/'))",
    '-DCMAKE_BUILD_TYPE=Release',
    '-DCMAKE_NINJA_FORCE_RESPONSE_FILE=ON',
    "-DCMAKE_C_COMPILER=$compilerCmake",
    "-DCMAKE_CXX_COMPILER=$compilerCmake",
    "-DCMAKE_CUDA_HOST_COMPILER=$compilerCmake",
    "-DCMAKE_CUDA_COMPILER=$cudaCmake/bin/nvcc.exe",
    "-DCUDAToolkit_ROOT=$cudaCmake",
    "-DCMAKE_CUDA_ARCHITECTURES=$architectures",
    "-DPython3_EXECUTABLE=$($python.Replace('\', '/'))",
    # Older FindCUDAToolkit modules only list lib/x64 on Windows.
    "-DCMAKE_LIBRARY_PATH=$cudaCmake/lib/arm64",
    "-DCUDA_CUDART=$cudaCmake/lib/arm64/cudart.lib",
    "-DCUDA_cudart_LIBRARY=$cudaCmake/lib/arm64/cudart.lib",
    "-DCUDA_cudart_static_LIBRARY=$cudaCmake/lib/arm64/cudart_static.lib",
    "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$buildCmake/bin",
    "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=$buildCmake/lib",
    "-DCMAKE_INSTALL_PREFIX=$($InstallPrefix.Replace('\', '/'))",
    '-DCUDARTLIB=cudart_static',
    '-DBUILD_NCCL4PY=OFF', '-DBUILD_PACKAGES=OFF',
    '-DRDMA_CORE=OFF', '-DMLX5DV=OFF',
    "-DNCCL_GIT_COMMIT_HASH=$expectedCommit"
)
$configure | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $BuildRoot 'configure-args.json')
Invoke-Checked $cmake $configure
Invoke-Checked $cmake @('--build', $BuildRoot, '--config', 'Release', '--target', 'nccl', '--parallel', "$Jobs", '--', '-k', '0')

$dll = Join-Path $BuildRoot 'bin/nccl.dll'
$importLibrary = Join-Path $BuildRoot 'lib/nccl.lib'
$generatedHeaders = Join-Path $BuildRoot 'include'
Assert-Arm64Pe $dll
Require-File $importLibrary
Require-File (Join-Path $generatedHeaders 'nccl.h')
Require-File (Join-Path $generatedHeaders 'nccl_device.h')
Require-File (Join-Path $generatedHeaders 'nccl_device/gin_win_stub.h')
$generatedHeader = [IO.File]::ReadAllText((Join-Path $generatedHeaders 'nccl.h'))
if ($generatedHeader -notmatch '(?m)^#define\s+NCCL_VERSION_CODE\s+23206\s*$') {
    throw 'The generated NCCL header does not declare version 23206'
}

# CMake's full install also requires nccl_static. Stage only the requested SDK.
foreach ($directory in @('include', 'lib', 'bin')) {
    New-Item -ItemType Directory -Force (Join-Path $InstallPrefix $directory) | Out-Null
}
Copy-Item -Path (Join-Path $generatedHeaders '*') -Destination (Join-Path $InstallPrefix 'include') -Recurse -Force
Copy-Item -LiteralPath $dll -Destination (Join-Path $InstallPrefix 'bin/nccl.dll') -Force
Copy-Item -LiteralPath $importLibrary -Destination (Join-Path $InstallPrefix 'lib/nccl.lib') -Force
foreach ($name in @('LICENSE.txt', 'LICENSE.nccl.txt')) {
    Copy-Item -LiteralPath (Join-Path $SourceRoot 'LICENSE.txt') -Destination (Join-Path $InstallPrefix $name) -Force
}
@{ cuda_nccl = $expectedVersion } | ConvertTo-Json -Compress |
    Set-Content -Encoding utf8 (Join-Path $InstallPrefix 'windows_cuda_versions.json')
@{
    source_commit = $actualCommit
    version = $expectedVersion
    cuda_architectures = $architectures
    host_compiler = $compiler
    cuda_compiler = $nvcc
    cmake = $cmake
    ninja = $ninja
    dll_sha256 = (Get-FileHash $dll -Algorithm SHA256).Hash.ToLowerInvariant()
    import_library_sha256 = (Get-FileHash $importLibrary -Algorithm SHA256).Hash.ToLowerInvariant()
} | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $InstallPrefix 'build-info.json')
Write-Output "Built NCCL $expectedVersion from $actualCommit for Windows ARM64: $InstallPrefix"
