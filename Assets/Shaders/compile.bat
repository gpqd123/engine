@echo off
:: 1. 设置 glslc.exe 的路径（如果它在环境变量里，直接写 glslc 即可）
set GLSLC=glslc.exe

:: 2. 创建输出文件夹
if not exist "spirv" mkdir "spirv"

echo [开始编译...]

:: 3. 循环编译当前目录下常见的 shader 后缀
for %%f in (*.vert *.frag *.comp *.geom) do (
    echo 正在处理: %%f
    "%GLSLC%" "%%f" -o "spirv/%%f.spv"
)

echo [编译完成]
pause