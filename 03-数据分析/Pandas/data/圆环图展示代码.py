# 圆环图展示结果
from pyecharts.charts import Pie
c = (
    Pie()
    .add(
        series_name="经验要求",
        data_pair=[list(z) for z in zip(
            job['exp'].value_counts().index.tolist(),# 准备数据
            job['exp'].value_counts().values.tolist()
        )],
        radius=["50%", "70%"],#圆环图，大环小环的半径大小
        label_opts=opts.LabelOpts(is_show=False, position="center"),
    )#设置图例位置
    .set_global_opts(
        title_opts=opts.TitleOpts(title="数据分析工作经验要求"),
        legend_opts=opts.LegendOpts(pos_left="right", orient="vertical"))
    .set_series_opts(
        tooltip_opts=opts.TooltipOpts(#鼠标滑过之后弹出文字格式
            trigger="item", 
            formatter="{a} <br/>{b}: {c} ({d}%)"
        ),
    )
)

c.render_notebook()