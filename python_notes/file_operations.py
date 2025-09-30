def write_to_file(filename, content):
    """
    向指定文件写入内容
    
    参数:
        filename (str): 文件名
        content (str): 要写入的内容
    """
    try:
        # 使用with语句打开文件，自动处理文件关闭
        # 'w'模式表示写入，如果文件不存在则创建，如果存在则覆盖
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"成功将内容写入文件: {filename}")
    except IOError as e: # IOError 是文件操作相关的错误，如权限问题、磁盘空间不足等
        print(f"写入文件时发生错误: {e}")

def read_from_file(filename):
    """
    从指定文件读取内容并返回
    
    参数:
        filename (str): 文件名
        
    返回:
        str: 文件内容，如果出错则返回None
    """
    try:
        # 'r'模式表示读取
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        print(f"成功从文件读取内容: {content}")
        return content
    except FileNotFoundError:
        print(f"错误: 文件 '{filename}' 不存在")
    except IOError as e:
        print(f"读取文件时发生错误: {e}")
    return None

def append_to_file(filename, content):
    """
    向指定文件追加内容
    
    参数:
        filename (str): 文件名
        content (str): 要追加的内容
    """
    try:
        # 'a'模式表示追加
        with open(filename, 'a', encoding='utf-8') as file:
            file.write(content)
        print(f"成功向文件追加内容: {filename}")
    except IOError as e:
        print(f"追加文件时发生错误: {e}")

if __name__ == "__main__":
    # 定义文件名
    demo_file = "demo_text.txt"
    
    # 写入初始内容
    initial_content = "这是第一行文本\n这是第二行文本\n"
    write_to_file(demo_file, initial_content)
    
    # 读取并打印内容
    print("\n--- 文件内容 ---")
    content = read_from_file(demo_file)
    if content:
        print(content)
    
    # 追加内容
    append_content = "这是追加的第三行文本\n"
    append_to_file(demo_file, append_content)
    
    # 再次读取并打印内容，查看追加效果
    print("\n--- 追加后的文件内容 ---")
    updated_content = read_from_file(demo_file)
    if updated_content:
        print(updated_content)
    