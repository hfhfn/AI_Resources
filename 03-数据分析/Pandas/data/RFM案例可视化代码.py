# 显示图形
from pyecharts.commons.utils import JsCode
from pyecharts import options as opts


range_color = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf',
               '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
range_max = int(display_data['number'].max())
c = (
    Bar3D()#设置了一个3D柱形图对象
    .add(
        "",#标题
        [d.tolist() for d in display_data.values],#数据
        xaxis3d_opts=opts.Axis3DOpts( type_="category",name='分组名称'),#x轴数据类型，名称
        yaxis3d_opts=opts.Axis3DOpts( type_="category",name='年份'),#y轴数据类型，名称
        zaxis3d_opts=opts.Axis3DOpts(type_="value",name='会员数量'),#z轴数据类型，名称
    )
    .set_global_opts(#设置颜色，及不同取值对应的颜色
        visualmap_opts=opts.VisualMapOpts(max_=range_max,range_color=range_color),
        title_opts=opts.TitleOpts(title="RFM分组结果"),#设置标题
    )
)
c.render_notebook() #在notebook中显示