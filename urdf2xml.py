from urdf2mjcf.convert import convert_urdf_to_mjcf # 直接导入 convert 函数
import os
import traceback

# 确保输出目录存在
if not os.path.exists("xml"):
    os.makedirs("xml")


try:
    urdf_path = "urdf/panda_arm.urdf"
    xml_output_path = "xml/panda_arm.xml"
    
    print(f"正在转换: {urdf_path} -> {xml_output_path}")
    
    # 直接调用 convert 函数
    convert_urdf_to_mjcf(urdf_path, xml_output_path)
    
    print("✅ 转换成功!")
    print(f"输出文件: {xml_output_path}")
    
except Exception as e:
    print(f"❌ 转换失败: {e}")
    traceback.print_exc() # 打印详细的错误堆栈
    print("请检查:")
    print("1. URDF文件是否存在")
    print("2. 是否安装了 urdf2mjcf 库: pip install urdf2mjcf")
    print("3. URDF文件格式是否正确")
