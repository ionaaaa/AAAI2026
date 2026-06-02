# 读取 sh 文件，按行分割，保留包含指定关键词的行

def filter_sh_lines(file_path, keywords):
    """
    读取 sh 文件，按行分割，只保留包含 keywords 中任意一个关键词的行。

    参数:
        file_path: str, sh 文件路径
        keywords: list[str], 关键词列表，行中只要包含其中任意一个关键词就保留

    返回:
        list[str], 保留的行列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 保留包含任意一个关键词的行
    filtered = [line for line in lines if any(kw in line for kw in keywords)]

    return filtered

# 示例用法:
keywords = ['sub-01']
result = filter_sh_lines('your_script.sh', keywords)
print(''.join(result))