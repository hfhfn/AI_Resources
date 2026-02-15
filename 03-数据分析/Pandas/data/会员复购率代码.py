def stats_reorder(start,end,col):
    """
    统计指定起始年月的复购率
    """
    #只要下单的数据  退单不统计
    order_data=custom_consume.query("订单类型=='下单'")
    #筛选日期
    order_data= order_data[(order_data['年月']<=end) & (order_data['年月']>=start)]
    #因为需要用到地区编号和年月  所以选择 订单日期  卡号   年月   地区编码  四个字段一起去重
    order_data=order_data[['订单日期','卡号','年月','地区编码']].drop_duplicates()
    #按照地区编码和卡号进行分组  统计订单日期数量  就是每个地区每个会员的购买次数
    consume_count = order_data.pivot_table(index =['地区编码','卡号'],values='订单日期',aggfunc='count').reset_index()
    #重命名列
    consume_count.rename(columns={'订单日期':'消费次数'},inplace=True)
    #判断是否复购
    consume_count['是否复购']=consume_count['消费次数']>1
    #统计每个地区的购买人数和复购人数
    depart_data=consume_count.pivot_table(index = ['地区编码'],values=['消费次数','是否复购'],aggfunc={'消费次数':'count','是否复购':'sum'})
    
    #重命名列
    depart_data.columns=['复购人数','购买人数']
    #计算复购率
    depart_data[col+'复购率']=depart_data['复购人数']/depart_data['购买人数']
   
    return depart_data