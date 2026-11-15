# Build Bazel 8.7's native Windows library with the ARM64 string-overread fix.
# SourceRoot must be Bazel 8.7.0, commit a6d8d66737d56766c0f5377c62ccf2ee562a1a47.
# This follows src/main/native/windows/build_windows_jni.sh, using ARM64 MSVC.
param(
    [Parameter(Mandatory = $true)][string]$SourceRoot,
    [Parameter(Mandatory = $true)][string]$OutputDirectory,
    [string]$JdkDirectory = $env:JAVA_HOME_21_AARCH64,
    [string]$VcDirectory = $env:BAZEL_VC
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$SourceRoot = (Resolve-Path $SourceRoot).Path
$JdkDirectory = (Resolve-Path $JdkDirectory).Path
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$OutputDirectory = (Resolve-Path $OutputDirectory).Path
$vcvars = Join-Path $VcDirectory 'Auxiliary/Build/vcvarsall.bat'
if (-not (Test-Path $vcvars)) { throw "Missing vcvarsall: $vcvars" }

# https://github.com/bazelbuild/bazel/pull/30606: JNI strings need an explicit length.
$jniUtil = Join-Path $SourceRoot 'src/main/native/windows/jni-util.cc'
$text = [IO.File]::ReadAllText($jniUtil)
$before = @'
    const jchar* jstr = env->GetStringChars(str, nullptr);
'@
$after = @'
    const jsize len = env->GetStringLength(str);
    const jchar* jstr = env->GetStringChars(str, nullptr);
'@
$assignBefore = '    result.assign(reinterpret_cast<const WCHAR*>(jstr));'
$assignAfter = @'
    result.assign(reinterpret_cast<const WCHAR*>(jstr),
                  static_cast<size_t>(len));
'@
if (-not $text.Contains($before) -or -not $text.Contains($assignBefore)) {
    throw 'The Bazel 8.7.0 GetJavaWstring source does not match the expected patch context'
}
$text = $text.Replace($before, $after).Replace($assignBefore, $assignAfter)
[IO.File]::WriteAllText($jniUtil, $text)

# These are normal JDK headers, copied as Bazel's native BUILD genrules do.
# No javac -h, generated Java classes, Bazel build, or third-party C++ libs needed.
$headers = Join-Path $OutputDirectory 'headers/src/main/native'
New-Item -ItemType Directory -Force $headers | Out-Null
Copy-Item (Join-Path $JdkDirectory 'include/jni.h') $headers
Copy-Item (Join-Path $JdkDirectory 'include/win32/jni_md.h') $headers
$headerRoot = Join-Path $OutputDirectory 'headers'

# Same source set as the upstream standalone Windows JNI bootstrap script.
$sources = @(
    'file-jni.cc', 'file.cc', 'jni-util.cc', 'process.cc', 'processes-jni.cc',
    'sleep_prevention_jni.cc', 'system_cpu_speed_monitor_jni.cc',
    'system_disk_space_monitor_jni.cc', 'system_load_advisory_monitor_jni.cc',
    'system_memory_pressure_jni.cc', 'system_network_stats.cc',
    'system_suspension_monitor_jni.cc', 'system_thermal_monitor_jni.cc', 'util.cc'
) | ForEach-Object {
    $path = Join-Path $SourceRoot "src/main/native/windows/$_"
    if (-not (Test-Path $path)) { throw "Missing Bazel JNI source: $path" }
    '"' + $path + '"'
}
# MSVC passes only the rest of the /link line to the linker.
$response = Join-Path $OutputDirectory 'compile.rsp'
@(
    '/nologo', '/O2', '/EHsc', '/LD', '/Fe:windows_jni.dll',
    "/I`"$headerRoot`"", "/I`"$SourceRoot`""
) + $sources + @('/link /DEFAULTLIB:advapi32.lib /MACHINE:ARM64') |
    Set-Content -Encoding ascii $response

$batch = Join-Path $OutputDirectory 'compile.cmd'
@(
    '@echo off',
    "call `"$vcvars`" arm64",
    'if errorlevel 1 exit /b %errorlevel%',
    "cd /d `"$OutputDirectory`"",
    "cl.exe @`"$response`"",
    'exit /b %errorlevel%'
) | Set-Content -Encoding ascii $batch
& cmd.exe /d /c $batch
if ($LASTEXITCODE -ne 0) { throw "JNI compilation failed: $LASTEXITCODE" }

$dll = Join-Path $OutputDirectory 'windows_jni.dll'
$bytes = [IO.File]::ReadAllBytes($dll)
$peOffset = [BitConverter]::ToInt32($bytes, 0x3c)
$machine = [BitConverter]::ToUInt16($bytes, $peOffset + 4)
if ($machine -ne 0xAA64) { throw "JNI library is not ARM64: machine=$machine" }
Write-Output "Built ARM64 JNI library: $dll"
Write-Output "Bazel startup option: --host_jvm_args=-Djava.library.path=$OutputDirectory"
