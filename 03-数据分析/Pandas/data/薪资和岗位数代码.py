# 将所有城市的薪资和就业岗位数合并到一起
def prepare_data(data):
    # 取出各城市有多少岗位
    temp1 = data.groupby('city')['url'].count().sort_values(ascending = False)
    # 计算出各城市岗位的平均起薪
    temp = data.groupby('city')['salary_down'].mean().sort_values(ascending = False)
    # 合并数据
    temp2 = pd.concat([temp,temp1],axis = 1)
    temp2 = temp2.reset_index()
    return temp2