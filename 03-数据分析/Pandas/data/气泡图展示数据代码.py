# 气泡图展示数据
from pyecharts.charts import Scatter
from pyecharts.commons.utils import JsCode

c = (
    Scatter() #创建散点图对象
    .add_xaxis(salary_data.salary_down.astype(int))#添加x周数据（薪资）
    .add_yaxis(
        "数据分析岗位数量", #y轴数据说明
        [list(z) for z in zip(salary_data.job_count, salary_data.city)],#Y轴数据，岗位，城市
        label_opts=opts.LabelOpts(#Js代码控制气泡显示提示文字
            formatter=JsCode(
                "function(params){return params.value[2]}" #提示
            )
        ),
    )
    .set_global_opts(#全局变量
        title_opts=opts.TitleOpts(title="数据分析就业岗位数量与平均起薪"),#设置标题
        tooltip_opts=opts.TooltipOpts(#Js代码控制气泡弹窗提示文字
            formatter=JsCode(
                "function (params) {return params.value[2]+ '平均薪资：'+params.value[0]}"
            )
        ),
        visualmap_opts=opts.VisualMapOpts(#控制
            type_="size", max_=1500, min_=200, dimension=1
        ),
        xaxis_opts=opts.AxisOpts(min_=6000,name='平均起薪'),#设置X轴起始值，X轴名字
        yaxis_opts=opts.AxisOpts(min_=300,max_=1550,name='岗位数量'),#设置Y轴起始值，Y轴名字
    )
)