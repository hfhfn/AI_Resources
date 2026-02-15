# 数据清洗 提取起薪
import re
def get_salary_down(x):
    if ('面议' in x) or ('小时' in x) or ('以' in x):
        return 0
    elif '元' in x:
        return int(re.search(pattern="([0-9]+)-([0-9]+)元/月", string=x).group(1))
    elif '-' in x:
        return int(re.search(pattern="([0-9]+)-([0-9]+)", string=x).group(1))
    else:
        return int(x)