-- 添加调试和发布模式规则
add_rules("mode.debug", "mode.release")

target("Engine")
    -- 编译为可执行文件
    set_kind("binary")
    
    -- 必须启用 C++23 以支持 <print> 和 std::print
    set_languages("cxx23") 
    
    -- 将项目根目录添加为头文件搜索路径
    -- 这样 main.cpp 中的 #include "Source/Runtime/Core/Application.hpp" 才能正确解析
    add_includedirs(".")
    
    -- 添加所有 cpp 参与编译 (递归搜索所有子目录)
    add_files("*.cpp")
    add_files("Source/**.cpp")