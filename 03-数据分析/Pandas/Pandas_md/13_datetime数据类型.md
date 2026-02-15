# 13 datetime 数据类型

## 学习目标

- 应用Pandas来处理日期时间类型数据

## 1 Python的datetime对象

- Python内置了datetime对象，可以在datetime库中找到

```python
from datetime import datetime
now = datetime.now()
now
```

><font color='red'>显示结果：</font>
>
>```shell
>datetime.datetime(2020, 6, 17, 19, 47, 56, 965416)
>```

- 还可以手动创建datetime

```python
t1 = datetime.now()
t2 = datetime(2020,1,1)
diff = t1-t2
print(diff)
```

><font color='red'>显示结果：</font>
>
>```shell
>168 days, 20:16:50.438044
>```

- 查看diff的数据类型

```python
print(type(diff))
```

><font color='red'>显示结果：</font>
>
>```shell
><class 'datetime.timedelta'>
>```

## 2 将pandas中的数据转换成datetime

- 可以使用to_datetime函数把数据转换成datetime类型

```python
#加载数据 并把Date列转换为datetime对象
ebola = pd.read_csv('data/country_timeseries.csv')
#获取左上角数据
ebola.iloc[:5,:5]
```

><font color='red'>显示结果：</font>
>
>|      |       Date |  Day | Cases_Guinea | Cases_Liberia | Cases_SierraLeone |
>| ---: | ---------: | ---: | -----------: | ------------: | ----------------: |
>|    0 |   1/5/2015 |  289 |       2776.0 |           NaN |           10030.0 |
>|    1 |   1/4/2015 |  288 |       2775.0 |           NaN |            9780.0 |
>|    2 |   1/3/2015 |  287 |       2769.0 |        8166.0 |            9722.0 |
>|    3 |   1/2/2015 |  286 |          NaN |        8157.0 |               NaN |
>|    4 | 12/31/2014 |  284 |       2730.0 |        8115.0 |            9633.0 |

- 从数据中看出 Date列是日期，但通过info查看加载后数据为object类型

```python
ebola.info()
```

><font color='red'>显示结果：</font>
>
>```shell
><class 'pandas.core.frame.DataFrame'>
>RangeIndex: 122 entries, 0 to 121
>Data columns (total 18 columns):
> #   Column               Non-Null Count  Dtype  
>---  ------               --------------  -----  
> 0   Date                 122 non-null    object 
> 1   Day                  122 non-null    int64  
> 2   Cases_Guinea         93 non-null     float64
> 3   Cases_Liberia        83 non-null     float64
> 4   Cases_SierraLeone    87 non-null     float64
> 5   Cases_Nigeria        38 non-null     float64
> 6   Cases_Senegal        25 non-null     float64
> 7   Cases_UnitedStates   18 non-null     float64
> 8   Cases_Spain          16 non-null     float64
> 9   Cases_Mali           12 non-null     float64
> 10  Deaths_Guinea        92 non-null     float64
> 11  Deaths_Liberia       81 non-null     float64
> 12  Deaths_SierraLeone   87 non-null     float64
> 13  Deaths_Nigeria       38 non-null     float64
> 14  Deaths_Senegal       22 non-null     float64
> 15  Deaths_UnitedStates  18 non-null     float64
> 16  Deaths_Spain         16 non-null     float64
> 17  Deaths_Mali          12 non-null     float64
>dtypes: float64(16), int64(1), object(1)
>memory usage: 17.3+ KB
>```

- 可以通过to_datetime方法把Date列转换为datetime,然后创建新列

```python
ebola['date_dt'] = pd.to_datetime(ebola['Date'])
ebola.info()
```

><font color='red'>显示结果：</font>
>
>```shell
><class 'pandas.core.frame.DataFrame'>
>RangeIndex: 122 entries, 0 to 121
>Data columns (total 19 columns):
> #   Column               Non-Null Count  Dtype         
>---  ------               --------------  -----         
> 0   Date                 122 non-null    object        
> 1   Day                  122 non-null    int64         
> 2   Cases_Guinea         93 non-null     float64       
> 3   Cases_Liberia        83 non-null     float64       
> 4   Cases_SierraLeone    87 non-null     float64       
> 5   Cases_Nigeria        38 non-null     float64       
> 6   Cases_Senegal        25 non-null     float64       
> 7   Cases_UnitedStates   18 non-null     float64       
> 8   Cases_Spain          16 non-null     float64       
> 9   Cases_Mali           12 non-null     float64       
> 10  Deaths_Guinea        92 non-null     float64       
> 11  Deaths_Liberia       81 non-null     float64       
> 12  Deaths_SierraLeone   87 non-null     float64       
> 13  Deaths_Nigeria       38 non-null     float64       
> 14  Deaths_Senegal       22 non-null     float64       
> 15  Deaths_UnitedStates  18 non-null     float64       
> 16  Deaths_Spain         16 non-null     float64       
> 17  Deaths_Mali          12 non-null     float64       
> 18  date_dt              122 non-null    datetime64[ns]
>dtypes: datetime64[ns](1), float64(16), int64(1), object(1)
>memory usage: 18.2+ KB
>```

- 如果数据中包含日期时间数据，可以在加载的时候，通过parse_dates参数指定

```python
ebola = pd.read_csv('data/country_timeseries.csv',parse_dates=[0])
ebola.info()
```

><font color='red'>显示结果：</font>
>
>```shell
><class 'pandas.core.frame.DataFrame'>
>RangeIndex: 122 entries, 0 to 121
>Data columns (total 18 columns):
> #   Column               Non-Null Count  Dtype         
>---  ------               --------------  -----         
> 0   Date                 122 non-null    datetime64[ns]
> 1   Day                  122 non-null    int64         
> 2   Cases_Guinea         93 non-null     float64       
> 3   Cases_Liberia        83 non-null     float64       
> 4   Cases_SierraLeone    87 non-null     float64       
> 5   Cases_Nigeria        38 non-null     float64       
> 6   Cases_Senegal        25 non-null     float64       
> 7   Cases_UnitedStates   18 non-null     float64       
> 8   Cases_Spain          16 non-null     float64       
> 9   Cases_Mali           12 non-null     float64       
> 10  Deaths_Guinea        92 non-null     float64       
> 11  Deaths_Liberia       81 non-null     float64       
> 12  Deaths_SierraLeone   87 non-null     float64       
> 13  Deaths_Nigeria       38 non-null     float64       
> 14  Deaths_Senegal       22 non-null     float64       
> 15  Deaths_UnitedStates  18 non-null     float64       
> 16  Deaths_Spain         16 non-null     float64       
> 17  Deaths_Mali          12 non-null     float64       
>dtypes: datetime64[ns](1), float64(16), int64(1)
>memory usage: 17.3 KB
>```

## 3 提取日期的各个部分

- 获取了一个datetime对象，就可以提取日期的各个部分了

```python
d = pd.to_datetime('2020-06-20')
d
```

><font color='red'>显示结果：</font>
>
>```shell
>Timestamp('2020-06-20 00:00:00')
>```

- 可以看到得到的数据是Timestamp类型，通过Timestamp可以获取年，月，日等部分

```python
d.year
```

><font color='red'>显示结果：</font>
>
>```
>2020
>```

```python
d.month
```

><font color='red'>显示结果：</font>
>
>```
>6
>```

```python
d.day
```

><font color='red'>显示结果：</font>
>
>```
>20
>```

- 通过ebola数据集的Date列，创建新列year

```python
ebola['year'] = ebola['Date'].dt.year
ebola['year']
```

><font color='red'>显示结果：</font>
>
>```shell
>0      2015
>1      2015
>2      2015
>3      2015
>4      2014
>       ... 
>117    2014
>118    2014
>119    2014
>120    2014
>121    2014
>Name: year, Length: 122, dtype: int64
>```

```python
ebola['month'],ebola['day']  = (ebola['Date'].dt.month,ebola['Date'].dt.day)
ebola[['Date','year','month','day']].head()
```

><font color='red'>显示结果：</font>
>
>|      |       Date | year | month |  day |
>| ---: | ---------: | ---: | ----: | ---: |
>|    0 | 2015-01-05 | 2015 |     1 |    5 |
>|    1 | 2015-01-04 | 2015 |     1 |    4 |
>|    2 | 2015-01-03 | 2015 |     1 |    3 |
>|    3 | 2015-01-02 | 2015 |     1 |    2 |
>|    4 | 2014-12-31 | 2014 |    12 |   31 |

```python
ebola.info()
```

><font color='red'>显示结果：</font>
>
>```shell
><class 'pandas.core.frame.DataFrame'>
>RangeIndex: 122 entries, 0 to 121
>Data columns (total 21 columns):
> #   Column               Non-Null Count  Dtype         
>---  ------               --------------  -----         
> 0   Date                 122 non-null    datetime64[ns]
> 1   Day                  122 non-null    int64         
> 2   Cases_Guinea         93 non-null     float64       
> 3   Cases_Liberia        83 non-null     float64       
> 4   Cases_SierraLeone    87 non-null     float64       
> 5   Cases_Nigeria        38 non-null     float64       
> 6   Cases_Senegal        25 non-null     float64       
> 7   Cases_UnitedStates   18 non-null     float64       
> 8   Cases_Spain          16 non-null     float64       
> 9   Cases_Mali           12 non-null     float64       
> 10  Deaths_Guinea        92 non-null     float64       
> 11  Deaths_Liberia       81 non-null     float64       
> 12  Deaths_SierraLeone   87 non-null     float64       
> 13  Deaths_Nigeria       38 non-null     float64       
> 14  Deaths_Senegal       22 non-null     float64       
> 15  Deaths_UnitedStates  18 non-null     float64       
> 16  Deaths_Spain         16 non-null     float64       
> 17  Deaths_Mali          12 non-null     float64       
> 18  year                 122 non-null    int64         
> 19  month                122 non-null    int64         
> 20  day                  122 non-null    int64         
>dtypes: datetime64[ns](1), float64(16), int64(4)
>memory usage: 20.1 KB
>```

## 4 日期运算和Timedelta

- Ebola数据集中的Day列表示一个国家爆发Ebola疫情的天数。这一列数据可以通过日期运算重建该列
- 疫情爆发的第一天（数据集中最早的一天）是2014-03-22。计算疫情爆发的天数时，只需要用每个日期减去这个日期即可

```python
# 获取疫情爆发的第一天
ebola['Date'].min()
```

><font color='red'>显示结果：</font>
>
>```
>Timestamp('2014-03-22 00:00:00')
>```

```python
ebola['outbreak_d'] = ebola['Date']-ebola['Date'].min()
ebola[['Date','Day','outbreak_d']].head()
```

><font color='red'>显示结果：</font>
>
>|      |       Date |  Day | outbreak_d |
>| ---: | ---------: | ---: | ---------: |
>|    0 | 2015-01-05 |  289 |   289 days |
>|    1 | 2015-01-04 |  288 |   288 days |
>|    2 | 2015-01-03 |  287 |   287 days |
>|    3 | 2015-01-02 |  286 |   286 days |
>|    4 | 2014-12-31 |  284 |   284 days |

```python
ebola[['Date','Day','outbreak_d']].tail()
```

><font color='red'>显示结果：</font>
>
>|      |       Date |  Day | outbreak_d |
>| ---: | ---------: | ---: | ---------: |
>|  117 | 2014-03-27 |    5 |     5 days |
>|  118 | 2014-03-26 |    4 |     4 days |
>|  119 | 2014-03-25 |    3 |     3 days |
>|  120 | 2014-03-24 |    2 |     2 days |
>|  121 | 2014-03-22 |    0 |     0 days |

- 执行这种日期运算，会得到一个timedelta对象

```python
ebola.info()
```

><font color='red'>显示结果：</font>
>
>```shell
><class 'pandas.core.frame.DataFrame'>
>RangeIndex: 122 entries, 0 to 121
>Data columns (total 22 columns):
> #   Column               Non-Null Count  Dtype          
>---  ------               --------------  -----          
> 0   Date                 122 non-null    datetime64[ns] 
> 1   Day                  122 non-null    int64          
> 2   Cases_Guinea         93 non-null     float64        
> 3   Cases_Liberia        83 non-null     float64        
> 4   Cases_SierraLeone    87 non-null     float64        
> 5   Cases_Nigeria        38 non-null     float64        
> 6   Cases_Senegal        25 non-null     float64        
> 7   Cases_UnitedStates   18 non-null     float64        
> 8   Cases_Spain          16 non-null     float64        
> 9   Cases_Mali           12 non-null     float64        
> 10  Deaths_Guinea        92 non-null     float64        
> 11  Deaths_Liberia       81 non-null     float64        
> 12  Deaths_SierraLeone   87 non-null     float64        
> 13  Deaths_Nigeria       38 non-null     float64        
> 14  Deaths_Senegal       22 non-null     float64        
> 15  Deaths_UnitedStates  18 non-null     float64        
> 16  Deaths_Spain         16 non-null     float64        
> 17  Deaths_Mali          12 non-null     float64        
> 18  year                 122 non-null    int64          
> 19  month                122 non-null    int64          
> 20  day                  122 non-null    int64          
> 21  outbreak_d           122 non-null    timedelta64[ns]
>dtypes: datetime64[ns](1), float64(16), int64(4), timedelta64[ns](1)
>memory usage: 21.1 KB
>```

- 案例

```python
# 加载数据
banks = pd.read_csv('data/banklist.csv')
banks.head()
```

><font color='red'>显示结果：</font>
>
>|      |                                         Bank Name |               City |   ST |  CERT |               Acquiring Institution | Closing Date | Updated Date |
>| ---: | ------------------------------------------------: | -----------------: | ---: | ----: | ----------------------------------: | -----------: | -----------: |
>|    0 |                               Fayette County Bank |         Saint Elmo |   IL |  1802 |           United Fidelity Bank, fsb |    26-May-17 |    26-Jul-17 |
>|    1 | Guaranty Bank, (d/b/a BestBank in Georgia & Mi... |          Milwaukee |   WI | 30003 | First-Citizens Bank & Trust Company |     5-May-17 |    26-Jul-17 |
>|    2 |                                    First NBC Bank |        New Orleans |   LA | 58302 |                        Whitney Bank |    28-Apr-17 |    26-Jul-17 |
>|    3 |                                     Proficio Bank | Cottonwood Heights |   UT | 35495 |                   Cache Valley Bank |     3-Mar-17 |    18-May-17 |
>|    4 |                     Seaway Bank and Trust Company |            Chicago |   IL | 19328 |                 State Bank of Texas |    27-Jan-17 |    18-May-17 |

- 可以看到有两列数据是日期时间类型，可以在导入数据的时候直接解析日期

```python
banks = pd.read_csv('data/banklist.csv',parse_dates=[5,6])
banks.info()
```

><font color='red'>显示结果：</font>
>
>```shell
><class 'pandas.core.frame.DataFrame'>
>RangeIndex: 553 entries, 0 to 552
>Data columns (total 7 columns):
> #   Column                 Non-Null Count  Dtype         
>---  ------                 --------------  -----         
> 0   Bank Name              553 non-null    object        
> 1   City                   553 non-null    object        
> 2   ST                     553 non-null    object        
> 3   CERT                   553 non-null    int64         
> 4   Acquiring Institution  553 non-null    object        
> 5   Closing Date           553 non-null    datetime64[ns]
> 6   Updated Date           553 non-null    datetime64[ns]
>dtypes: datetime64[ns](2), int64(1), object(4)
>memory usage: 30.4+ KB
>```

- 添加两列，分别表示银行破产的季度和年份

```python
banks['closing_quarter'],banks['closing_year'] = (banks['Closing Date'].dt.quarter,banks['Closing Date'].dt.year)
```

- 可以根据新添加的两列，计算每年破产银行数量，计算每年每季度破产银行数量

```python
closing_year = banks.groupby(['closing_year']).size()
closing_year
```

><font color='red'>显示结果：</font>
>
>```shell
>closing_year
>2000      2
>2001      4
>2002     11
>2003      3
>2004      4
>2007      3
>2008     25
>2009    140
>2010    157
>2011     92
>2012     51
>2013     24
>2014     18
>2015      8
>2016      5
>2017      6
>dtype: int64
>```

```python
closing_year_q = banks.groupby(['closing_year','closing_quarter']).size()
```

- 可以使用绘图函数绘制结果

```python
closing_year.plot()
```

><font color='red'>显示结果：</font>
>
>![img](data:image/png;base64,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)

```python
closing_year_q.plot()
```

><font color='red'>显示结果：</font>
>
>![img](data:image/png;base64,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)

## 5 处理股票数据

- 股票价格是包含日期的典型数据

```python
#加载特斯拉股票数据
tesla = pd.read_csv('data/TSLA.csv')
tesla
```

><font color='red'>显示结果：</font>
>
>|      |       Date |        High |        Low |        Open |       Close |   Volume |   Adj Close |
>| ---: | ---------: | ----------: | ---------: | ----------: | ----------: | -------: | ----------: |
>|    0 | 2015-08-19 |  260.649994 | 255.020004 |  260.329987 |  255.250000 |  3604300 |  255.250000 |
>|    1 | 2015-08-20 |  254.559998 | 241.899994 |  252.059998 |  242.179993 |  4905800 |  242.179993 |
>|    2 | 2015-08-21 |  243.800003 | 230.509995 |  236.000000 |  230.770004 |  6590200 |  230.770004 |
>|    3 | 2015-08-24 |  231.399994 | 195.000000 |  202.789993 |  218.869995 |  9581600 |  218.869995 |
>|    4 | 2015-08-25 |  230.899994 | 219.119995 |  230.520004 |  220.029999 |  4327300 |  220.029999 |
>|  ... |        ... |         ... |        ... |         ... |         ... |      ... |         ... |
>| 1210 | 2020-06-10 | 1027.479980 | 982.500000 |  991.880005 | 1025.050049 | 18563400 | 1025.050049 |
>| 1211 | 2020-06-11 | 1018.960022 | 972.000000 |  990.200012 |  972.840027 | 15916500 |  972.840027 |
>| 1212 | 2020-06-12 |  987.979980 | 912.599976 |  980.000000 |  935.280029 | 16730200 |  935.280029 |
>| 1213 | 2020-06-15 |  998.840027 | 908.500000 |  917.789978 |  990.900024 | 15697200 |  990.900024 |
>| 1214 | 2020-06-16 | 1012.880005 | 962.390015 | 1011.849976 |  982.130005 | 14051100 |  982.130005 |
>
>1215 rows × 7 columns

- 可以看出，tesla股票数据中第一列为日期，在加载数据的时候，可以直接解析日期数据

```python
tesla = pd.read_csv('data/TSLA.csv',parse_dates=[0])
tesla.info()
```

><font color='red'>显示结果：</font>
>
>```shell
><class 'pandas.core.frame.DataFrame'>
>RangeIndex: 1215 entries, 0 to 1214
>Data columns (total 7 columns):
> #   Column     Non-Null Count  Dtype         
>---  ------     --------------  -----         
> 0   Date       1215 non-null   datetime64[ns]
> 1   High       1215 non-null   float64       
> 2   Low        1215 non-null   float64       
> 3   Open       1215 non-null   float64       
> 4   Close      1215 non-null   float64       
> 5   Volume     1215 non-null   int64         
> 6   Adj Close  1215 non-null   float64       
>dtypes: datetime64[ns](1), float64(5), int64(1)
>memory usage: 66.6 KB
>```

### 5.1 基于日期数据获取数据子集

```python
#获取2015年8月的股票数据
tesla.loc[(tesla.Date.dt.year ==2015) & (tesla.Date.dt.month==8)]
```

><font color='red'>显示结果：</font>
>
>|      |       Date |       High |        Low |       Open |      Close |  Volume |  Adj Close |
>| ---: | ---------: | ---------: | ---------: | ---------: | ---------: | ------: | ---------: |
>|    0 | 2015-08-19 | 260.649994 | 255.020004 | 260.329987 | 255.250000 | 3604300 | 255.250000 |
>|    1 | 2015-08-20 | 254.559998 | 241.899994 | 252.059998 | 242.179993 | 4905800 | 242.179993 |
>|    2 | 2015-08-21 | 243.800003 | 230.509995 | 236.000000 | 230.770004 | 6590200 | 230.770004 |
>|    3 | 2015-08-24 | 231.399994 | 195.000000 | 202.789993 | 218.869995 | 9581600 | 218.869995 |
>|    4 | 2015-08-25 | 230.899994 | 219.119995 | 230.520004 | 220.029999 | 4327300 | 220.029999 |
>|    5 | 2015-08-26 | 228.000000 | 215.509995 | 227.929993 | 224.839996 | 4963000 | 224.839996 |
>|    6 | 2015-08-27 | 244.750000 | 230.809998 | 231.000000 | 242.990005 | 7656000 | 242.990005 |
>|    7 | 2015-08-28 | 251.449997 | 241.570007 | 241.860001 | 248.479996 | 5513700 | 248.479996 |
>|    8 | 2015-08-31 | 254.949997 | 245.509995 | 245.619995 | 249.059998 | 4700200 | 249.059998 |

#### DatetimeIndex对象

- 在处理包含datetime的数据时，经常需要把datetime对象设置成DateFrame的索引

```python
#首先把Date列指定为索引
tesla.index = tesla['Date']
tesla.index
```

><font color='red'>显示结果：</font>
>
>```shell
>DatetimeIndex(['2015-08-19', '2015-08-20', '2015-08-21', '2015-08-24',
>               '2015-08-25', '2015-08-26', '2015-08-27', '2015-08-28',
>               '2015-08-31', '2015-09-01',
>               ...
>               '2020-06-03', '2020-06-04', '2020-06-05', '2020-06-08',
>               '2020-06-09', '2020-06-10', '2020-06-11', '2020-06-12',
>               '2020-06-15', '2020-06-16'],
>              dtype='datetime64[ns]', name='Date', length=1215, freq=None)
>```

```python
tesla.head()
```

><font color='red'>显示结果：</font>
>
>|            |       Date |       High |        Low |       Open |      Close |  Volume |  Adj Close |
>| ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ------: | ---------: |
>|       Date |            |            |            |            |            |         |            |
>| 2015-08-19 | 2015-08-19 | 260.649994 | 255.020004 | 260.329987 | 255.250000 | 3604300 | 255.250000 |
>| 2015-08-20 | 2015-08-20 | 254.559998 | 241.899994 | 252.059998 | 242.179993 | 4905800 | 242.179993 |
>| 2015-08-21 | 2015-08-21 | 243.800003 | 230.509995 | 236.000000 | 230.770004 | 6590200 | 230.770004 |
>| 2015-08-24 | 2015-08-24 | 231.399994 | 195.000000 | 202.789993 | 218.869995 | 9581600 | 218.869995 |
>| 2015-08-25 | 2015-08-25 | 230.899994 | 219.119995 | 230.520004 | 220.029999 | 4327300 | 220.029999 |

- 把索引设置为日期对象后，可以直接使用日期来获取某些数据

```python
tesla['2016'].iloc[:5]
```

><font color='red'>显示结果：</font>
>
>|            |       Date |       High |        Low |       Open |      Close |  Volume |  Adj Close |
>| ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ------: | ---------: |
>|       Date |            |            |            |            |            |         |            |
>| 2016-01-04 | 2016-01-04 | 231.380005 | 219.000000 | 230.720001 | 223.410004 | 6827100 | 223.410004 |
>| 2016-01-05 | 2016-01-05 | 226.889999 | 220.000000 | 226.360001 | 223.429993 | 3186800 | 223.429993 |
>| 2016-01-06 | 2016-01-06 | 220.050003 | 215.979996 | 220.000000 | 219.039993 | 3779100 | 219.039993 |
>| 2016-01-07 | 2016-01-07 | 218.440002 | 213.669998 | 214.190002 | 215.649994 | 3554300 | 215.649994 |
>| 2016-01-08 | 2016-01-08 | 220.440002 | 210.770004 | 217.860001 | 211.000000 | 3628100 | 211.000000 |

- 也可以根据年份和月份获取数据

```python
tesla['2016-06'].iloc[:5]
```

><font color='red'>显示结果：</font>
>
>|            |       Date |       High |        Low |       Open |      Close |  Volume |  Adj Close |
>| ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | ------: | ---------: |
>|       Date |            |            |            |            |            |         |            |
>| 2016-06-01 | 2016-06-01 | 222.399994 | 216.889999 | 221.479996 | 219.559998 | 2982700 | 219.559998 |
>| 2016-06-02 | 2016-06-02 | 219.910004 | 217.110001 | 219.589996 | 218.960007 | 2032800 | 218.960007 |
>| 2016-06-03 | 2016-06-03 | 221.940002 | 218.009995 | 220.000000 | 218.990005 | 2229000 | 218.990005 |
>| 2016-06-06 | 2016-06-06 | 220.899994 | 215.449997 | 218.000000 | 220.679993 | 2249500 | 220.679993 |
>| 2016-06-07 | 2016-06-07 | 234.440002 | 221.520004 | 222.240005 | 232.339996 | 6213600 | 232.339996 |

#### TimedeltaIndex对象

- 首先创建一个timedelta

```python
tesla['ref_date'] = tesla['Date']-tesla['Date'].min()
tesla['ref_date'] 
```

><font color='red'>显示结果：</font>
>
>```shell
>Date
>2015-08-19      0 days
>2015-08-20      1 days
>2015-08-21      2 days
>2015-08-24      5 days
>2015-08-25      6 days
>                ...   
>2020-06-10   1757 days
>2020-06-11   1758 days
>2020-06-12   1759 days
>2020-06-15   1762 days
>2020-06-16   1763 days
>Name: ref_date, Length: 1215, dtype: timedelta64[ns]
>```

- 把timedelta设置为index

```python
tesla.index = tesla['ref_date'] 
tesla.info()
```

><font color='red'>显示结果：</font>
>
>```shell
><class 'pandas.core.frame.DataFrame'>
>TimedeltaIndex: 1215 entries, 0 days to 1763 days
>Data columns (total 8 columns):
> #   Column     Non-Null Count  Dtype          
>---  ------     --------------  -----          
> 0   Date       1215 non-null   datetime64[ns] 
> 1   High       1215 non-null   float64        
> 2   Low        1215 non-null   float64        
> 3   Open       1215 non-null   float64        
> 4   Close      1215 non-null   float64        
> 5   Volume     1215 non-null   int64          
> 6   Adj Close  1215 non-null   float64        
> 7   ref_date   1215 non-null   timedelta64[ns]
>dtypes: datetime64[ns](1), float64(5), int64(1), timedelta64[ns](1)
>memory usage: 85.4 KB
>```

```python
tesla.head()
```

><font color='red'>显示结果：</font>
>
>|          |       Date |       High |        Low |       Open |      Close |  Volume |  Adj Close | ref_date |
>| -------: | ---------: | ---------: | ---------: | ---------: | ---------: | ------: | ---------: | -------: |
>| ref_date |            |            |            |            |            |         |            |          |
>|   0 days | 2015-08-19 | 260.649994 | 255.020004 | 260.329987 | 255.250000 | 3604300 | 255.250000 |   0 days |
>|   1 days | 2015-08-20 | 254.559998 | 241.899994 | 252.059998 | 242.179993 | 4905800 | 242.179993 |   1 days |
>|   2 days | 2015-08-21 | 243.800003 | 230.509995 | 236.000000 | 230.770004 | 6590200 | 230.770004 |   2 days |
>|   5 days | 2015-08-24 | 231.399994 | 195.000000 | 202.789993 | 218.869995 | 9581600 | 218.869995 |   5 days |
>|   6 days | 2015-08-25 | 230.899994 | 219.119995 | 230.520004 | 220.029999 | 4327300 | 220.029999 |   6 days |

- 可以基于ref_date来选择数据

```python
tesla['0 days':'4 days']
```

><font color='red'>显示结果：</font>
>
>|          |       Date |       High |        Low |       Open |      Close |  Volume |  Adj Close | ref_date |
>| -------: | ---------: | ---------: | ---------: | ---------: | ---------: | ------: | ---------: | -------: |
>| ref_date |            |            |            |            |            |         |            |          |
>|   0 days | 2015-08-19 | 260.649994 | 255.020004 | 260.329987 | 255.250000 | 3604300 | 255.250000 |   0 days |
>|   1 days | 2015-08-20 | 254.559998 | 241.899994 | 252.059998 | 242.179993 | 4905800 | 242.179993 |   1 days |
>|   2 days | 2015-08-21 | 243.800003 | 230.509995 | 236.000000 | 230.770004 | 6590200 | 230.770004 |   2 days |

## 6 日期范围

- 包含日期的数据集中，并非每一个都包含固定频率。比如在Ebola数据集中，日期并没有规律

```python
ebola.iloc[:,:5]
```

><font color='red'>显示结果：</font>
>
>|      |       Date |  Day | Cases_Guinea | Cases_Liberia | Cases_SierraLeone |
>| ---: | ---------: | ---: | -----------: | ------------: | ----------------: |
>|    0 | 2015-01-05 |  289 |       2776.0 |           NaN |           10030.0 |
>|    1 | 2015-01-04 |  288 |       2775.0 |           NaN |            9780.0 |
>|    2 | 2015-01-03 |  287 |       2769.0 |        8166.0 |            9722.0 |
>|    3 | 2015-01-02 |  286 |          NaN |        8157.0 |               NaN |
>|    4 | 2014-12-31 |  284 |       2730.0 |        8115.0 |            9633.0 |
>|  ... |        ... |  ... |          ... |           ... |               ... |
>|  117 | 2014-03-27 |    5 |        103.0 |           8.0 |               6.0 |
>|  118 | 2014-03-26 |    4 |         86.0 |           NaN |               NaN |
>|  119 | 2014-03-25 |    3 |         86.0 |           NaN |               NaN |
>|  120 | 2014-03-24 |    2 |         86.0 |           NaN |               NaN |
>|  121 | 2014-03-22 |    0 |         49.0 |           NaN |               NaN |

- 从上面的数据中可以看到，缺少2015年1月1日，2014年3月23日，如果想让日期连续，可以创建一个日期范围来为数据集重建索引。
- 可以使用date_range函数来创建连续的日期范围

```python
head_range = pd.date_range(start='2014-12-31',end='2015-01-05')
head_range
```

><font color='red'>显示结果：</font>
>
>```shell
>DatetimeIndex(['2014-12-31', '2015-01-01', '2015-01-02', '2015-01-03',
>               '2015-01-04', '2015-01-05'],
>              dtype='datetime64[ns]', freq='D')
>```

- 在上面的例子中，只取前5行，首先设置日期索引，然后为数据重建连续索引

```python
ebola_5 = ebola.head()
ebola_5.index = ebola_5['Date']
ebola_5.reindex(head_range).iloc[:,:5]
```

><font color='red'>显示结果：</font>
>
>|            |       Date |   Day | Cases_Guinea | Cases_Liberia | Cases_SierraLeone |
>| ---------: | ---------: | ----: | -----------: | ------------: | ----------------: |
>| 2014-12-31 | 2014-12-31 | 284.0 |       2730.0 |        8115.0 |            9633.0 |
>| 2015-01-01 |        NaT |   NaN |          NaN |           NaN |               NaN |
>| 2015-01-02 | 2015-01-02 | 286.0 |          NaN |        8157.0 |               NaN |
>| 2015-01-03 | 2015-01-03 | 287.0 |       2769.0 |        8166.0 |            9722.0 |
>| 2015-01-04 | 2015-01-04 | 288.0 |       2775.0 |           NaN |            9780.0 |
>| 2015-01-05 | 2015-01-05 | 289.0 |       2776.0 |           NaN |           10030.0 |

- 使用date_range函数创建日期序列时，可以传入一个参数freq，默认情况下freq取值为D，表示日期范围内的值是逐日递增的。

```python
# 2020年1月1日这周所有的工作日
pd.date_range('2020-01-01','2020-01-07',freq='B')
```

><font color='red'>显示结果：</font>
>
>```shell
>DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-06',
>               '2020-01-07'],
>              dtype='datetime64[ns]', freq='B')
>```

- 从结果中看到生成的日期中缺少1月4日，1月5日，为休息日

<center>feq参数的可能取值</center>

| Alias    | Description                     |
| :------- | :------------------------------ |
| B        | 工作日                          |
| C        | 自定义工作日                    |
| D        | 日历日                          |
| W        | 每周                            |
| M        | 月末                            |
| SM       | 月中和月末（每月第15天和月末）  |
| BM       | 月末工作日                      |
| CBM      | 自定义月末工作日                |
| MS       | 月初                            |
| SMS      | 月初和月中（每月第1天和第15天） |
| BMS      | 月初工作日                      |
| CBMS     | 自定义月初工作日                |
| Q        | 季度末                          |
| BQ       | 季度末工作日                    |
| QS       | 季度初                          |
| BQS      | 季度初工作日                    |
| A, Y     | 年末                            |
| BA, BY   | 年末工作日                      |
| AS, YS   | 年初                            |
| BAS, BYS | 年初工作日                      |
| BH       | 工作时间                        |
| H        | 小时                            |
| T, min   | 分钟                            |
| S        | 秒                              |
| L, ms    | 毫秒                            |
| U, us    | microseconds                    |
| N        | 纳秒                            |

- 在freq传入参数的基础上，可以做一些调整

```python
# 隔一个工作日取一个工作日
pd.date_range('2020-01-01','2020-01-07',freq='2B')
```

><font color='red'>显示结果：</font>
>
>```shell
>DatetimeIndex(['2020-01-01', '2020-01-03', '2020-01-07'], dtype='datetime64[ns]', freq='2B')
>```

- freq传入的参数可以传入多个

```python
#2020年每个月的第一个星期四
pd.date_range('2020-01-01','2020-12-31',freq='WOM-1THU')
```

><font color='red'>显示结果：</font>
>
>```shell
>DatetimeIndex(['2020-01-02', '2020-02-06', '2020-03-05', '2020-04-02',
>               '2020-05-07', '2020-06-04', '2020-07-02', '2020-08-06',
>               '2020-09-03', '2020-10-01', '2020-11-05', '2020-12-03'],
>              dtype='datetime64[ns]', freq='WOM-1THU')
>```

```python
#每个月的第三个星期五
pd.date_range('2020-01-01','2020-12-31',freq='WOM-3FRI')
```

><font color='red'>显示结果：</font>
>
>```
>DatetimeIndex(['2020-01-17', '2020-02-21', '2020-03-20', '2020-04-17',
>               '2020-05-15', '2020-06-19', '2020-07-17', '2020-08-21',
>               '2020-09-18', '2020-10-16', '2020-11-20', '2020-12-18'],
>              dtype='datetime64[ns]', freq='WOM-3FRI')
>```

## 7 datetime类型案例

- 加载丹佛市犯罪记录数据集

```python
crime = pd.read_csv('data/crime.csv',parse_dates=['REPORTED_DATE'])
crime
```

><font color='red'>显示结果：</font>
>
>|        |              OFFENSE_TYPE_ID | OFFENSE_CATEGORY_ID |       REPORTED_DATE |     GEO_LON |   GEO_LAT |           NEIGHBORHOOD_ID | IS_CRIME | IS_TRAFFIC |
>| -----: | ---------------------------: | ------------------: | ------------------: | ----------: | --------: | ------------------------: | -------: | ---------: |
>|      0 |    traffic-accident-dui-duid |    traffic-accident | 2014-06-29 02:01:00 | -105.000149 | 39.745753 |                       cbd |        0 |          1 |
>|      1 |   vehicular-eluding-no-chase |    all-other-crimes | 2014-06-29 01:54:00 | -104.884660 | 39.738702 |               east-colfax |        1 |          0 |
>|      2 |         disturbing-the-peace |     public-disorder | 2014-06-29 02:00:00 | -105.020719 | 39.706674 |               athmar-park |        1 |          0 |
>|      3 |                       curfew |     public-disorder | 2014-06-29 02:18:00 | -105.001552 | 39.769505 |                 sunnyside |        1 |          0 |
>|      4 |           aggravated-assault |  aggravated-assault | 2014-06-29 04:17:00 | -105.018557 | 39.679229 | college-view-south-platte |        1 |          0 |
>|    ... |                          ... |                 ... |                 ... |         ... |       ... |                       ... |      ... |        ... |
>| 460906 |   burglary-business-by-force |            burglary | 2017-09-13 05:48:00 | -105.033840 | 39.762365 |             west-highland |        1 |          0 |
>| 460907 | weapon-unlawful-discharge-of |    all-other-crimes | 2017-09-12 20:37:00 | -105.040313 | 39.721264 |               barnum-west |        1 |          0 |
>| 460908 |       traf-habitual-offender |    all-other-crimes | 2017-09-12 16:32:00 | -104.847024 | 39.779596 |                 montbello |        1 |          0 |
>| 460909 |      criminal-mischief-other |     public-disorder | 2017-09-12 13:04:00 | -104.949182 | 39.756353 |                   skyland |        1 |          0 |
>| 460910 |                  theft-other |             larceny | 2017-09-12 09:30:00 | -104.985739 | 39.735045 |              capitol-hill |        1 |          0 |

```python
crime.info()
```

><font color='red'>显示结果：</font>
>
>```shell
><class 'pandas.core.frame.DataFrame'>
>RangeIndex: 460911 entries, 0 to 460910
>Data columns (total 8 columns):
> #   Column               Non-Null Count   Dtype         
>---  ------               --------------   -----         
> 0   OFFENSE_TYPE_ID      460911 non-null  object        
> 1   OFFENSE_CATEGORY_ID  460911 non-null  object        
> 2   REPORTED_DATE        460911 non-null  datetime64[ns]
> 3   GEO_LON              457296 non-null  float64       
> 4   GEO_LAT              457296 non-null  float64       
> 5   NEIGHBORHOOD_ID      460911 non-null  object        
> 6   IS_CRIME             460911 non-null  int64         
> 7   IS_TRAFFIC           460911 non-null  int64         
>dtypes: datetime64[ns](1), float64(2), int64(2), object(3)
>memory usage: 28.1+ MB
>```

```python
#设置报警时间为索引
crime = crime.set_index('REPORTED_DATE')
crime.head()
```

- 查看某一天的报警记录

```python
crime.loc['2016-05-12']
```

><font color='red'>显示结果：</font>
>
>|     OFFENSE_TYPE_ID |     OFFENSE_CATEGORY_ID |                      GEO_LON |     GEO_LAT | NEIGHBORHOOD_ID |            IS_CRIME | IS_TRAFFIC |      |
>| ------------------: | ----------------------: | ---------------------------: | ----------: | --------------: | ------------------: | ---------: | ---: |
>|       REPORTED_DATE |                         |                              |             |                 |                     |            |      |
>| 2016-05-12 23:51:00 | criminal-mischief-other |              public-disorder | -105.017241 |       39.705845 |         athmar-park |          1 |    0 |
>| 2016-05-12 18:40:00 |       liquor-possession |                 drug-alcohol | -104.995692 |       39.747875 |                 cbd |          1 |    0 |
>| 2016-05-12 22:26:00 |        traffic-accident |             traffic-accident | -104.880037 |       39.777037 |           stapleton |          0 |    1 |
>| 2016-05-12 20:35:00 |           theft-bicycle |                      larceny | -104.929350 |       39.763797 | northeast-park-hill |          1 |    0 |
>| 2016-05-12 09:39:00 |  theft-of-motor-vehicle |                   auto-theft | -104.941233 |       39.775510 |      elyria-swansea |          1 |    0 |
>|                 ... |                     ... |                          ... |         ... |             ... |                 ... |        ... |  ... |
>| 2016-05-12 17:55:00 |      public-peace-other |              public-disorder | -105.027747 |       39.700029 |            westwood |          1 |    0 |
>| 2016-05-12 19:24:00 |       threats-to-injure |              public-disorder | -104.947118 |       39.763777 |             clayton |          1 |    0 |
>| 2016-05-12 22:28:00 |           sex-aslt-rape |               sexual-assault |         NaN |             NaN |   harvey-park-south |          1 |    0 |
>| 2016-05-12 15:59:00 |  menacing-felony-w-weap |           aggravated-assault | -104.935172 |       39.723703 |             hilltop |          1 |    0 |
>| 2016-05-12 16:39:00 |              assault-dv | other-crimes-against-persons | -104.974700 |       39.740555 |  north-capitol-hill |          1 |    0 |
>
>243 rows × 7 columns

- 查看某一段时间的犯罪记录

```python
crime.loc['2015-3-4':'2016-1-1'].sort_index()
```

><font color='red'>显示结果：</font>
>
>|                     |              OFFENSE_TYPE_ID |          OFFENSE_CATEGORY_ID |     GEO_LON |   GEO_LAT |      NEIGHBORHOOD_ID | IS_CRIME | IS_TRAFFIC |
>| ------------------: | ---------------------------: | ---------------------------: | ----------: | --------: | -------------------: | -------: | ---------: |
>|       REPORTED_DATE |                              |                              |             |           |                      |          |            |
>| 2015-03-04 00:11:00 |                   assault-dv | other-crimes-against-persons | -105.021966 | 39.770883 |            sunnyside |        1 |          0 |
>| 2015-03-04 00:19:00 |                   assault-dv | other-crimes-against-persons | -104.978988 | 39.748799 |          five-points |        1 |          0 |
>| 2015-03-04 00:27:00 |            theft-of-services |                      larceny | -105.055082 | 39.790564 |                regis |        1 |          0 |
>| 2015-03-04 00:49:00 | traffic-accident-hit-and-run |             traffic-accident | -104.987454 | 39.701378 | washington-park-west |        0 |          1 |
>| 2015-03-04 01:07:00 |   burglary-business-no-force |                     burglary | -105.010843 | 39.762538 |             highland |        1 |          0 |
>|                 ... |                          ... |                          ... |         ... |       ... |                  ... |      ... |        ... |
>| 2016-01-01 23:15:00 | traffic-accident-hit-and-run |             traffic-accident | -104.996861 | 39.738612 |         civic-center |        0 |          1 |
>| 2016-01-01 23:16:00 |             traffic-accident |             traffic-accident | -105.025088 | 39.707590 |             westwood |        0 |          1 |
>| 2016-01-01 23:40:00 |             robbery-business |                      robbery | -105.039236 | 39.726157 |           villa-park |        1 |          0 |
>| 2016-01-01 23:45:00 |         drug-cocaine-possess |                 drug-alcohol | -104.987310 | 39.753598 |          five-points |        1 |          0 |
>| 2016-01-01 23:48:00 |      drug-poss-paraphernalia |                 drug-alcohol | -104.986020 | 39.752541 |          five-points |        1 |          0 |
>
>75403 rows × 7 columns

- 时间段可以包括小时分钟

```python
crime.loc['2015-3-4 22':'2016-1-1 23:45:00'].sort_index()
```

><font color='red'>显示结果：</font>
>
>|                     |              OFFENSE_TYPE_ID | OFFENSE_CATEGORY_ID |     GEO_LON |   GEO_LAT |      NEIGHBORHOOD_ID | IS_CRIME | IS_TRAFFIC |
>| ------------------: | ---------------------------: | ------------------: | ----------: | --------: | -------------------: | -------: | ---------: |
>|       REPORTED_DATE |                              |                     |             |           |                      |          |            |
>| 2015-03-04 22:25:00 | traffic-accident-hit-and-run |    traffic-accident | -104.973896 | 39.769064 |          five-points |        0 |          1 |
>| 2015-03-04 22:30:00 |             traffic-accident |    traffic-accident | -104.906412 | 39.632816 |        hampden-south |        0 |          1 |
>| 2015-03-04 22:32:00 | traffic-accident-hit-and-run |    traffic-accident | -104.979180 | 39.706613 | washington-park-west |        0 |          1 |
>| 2015-03-04 22:33:00 | traffic-accident-hit-and-run |    traffic-accident | -104.991655 | 39.740067 |         civic-center |        0 |          1 |
>| 2015-03-04 22:36:00 |      theft-unauth-use-of-ftd |  white-collar-crime | -105.045234 | 39.667928 |          harvey-park |        1 |          0 |
>|                 ... |                          ... |                 ... |         ... |       ... |                  ... |      ... |        ... |
>| 2016-01-01 23:07:00 |                   traf-other |    all-other-crimes | -104.980400 | 39.740144 |   north-capitol-hill |        1 |          0 |
>| 2016-01-01 23:15:00 | traffic-accident-hit-and-run |    traffic-accident | -104.996861 | 39.738612 |         civic-center |        0 |          1 |
>| 2016-01-01 23:16:00 |             traffic-accident |    traffic-accident | -105.025088 | 39.707590 |             westwood |        0 |          1 |
>| 2016-01-01 23:40:00 |             robbery-business |             robbery | -105.039236 | 39.726157 |           villa-park |        1 |          0 |
>| 2016-01-01 23:45:00 |         drug-cocaine-possess |        drug-alcohol | -104.987310 | 39.753598 |          five-points |        1 |          0 |
>
>75175 rows × 7 columns

- 查询凌晨两点到五点的报警记录

```python
crime.between_time('2:00', '5:00', include_end=False)
```

><font color='red'>显示结果：</font>
>
>|                     |                OFFENSE_TYPE_ID |          OFFENSE_CATEGORY_ID |     GEO_LON |   GEO_LAT |           NEIGHBORHOOD_ID | IS_CRIME | IS_TRAFFIC |
>| ------------------: | -----------------------------: | ---------------------------: | ----------: | --------: | ------------------------: | -------: | ---------: |
>|       REPORTED_DATE |                                |                              |             |           |                           |          |            |
>| 2014-06-29 02:01:00 |      traffic-accident-dui-duid |             traffic-accident | -105.000149 | 39.745753 |                       cbd |        0 |          1 |
>| 2014-06-29 02:00:00 |           disturbing-the-peace |              public-disorder | -105.020719 | 39.706674 |               athmar-park |        1 |          0 |
>| 2014-06-29 02:18:00 |                         curfew |              public-disorder | -105.001552 | 39.769505 |                 sunnyside |        1 |          0 |
>| 2014-06-29 04:17:00 |             aggravated-assault |           aggravated-assault | -105.018557 | 39.679229 | college-view-south-platte |        1 |          0 |
>| 2014-06-29 04:22:00 | violation-of-restraining-order |             all-other-crimes | -104.972447 | 39.739449 |             cheesman-park |        1 |          0 |
>|                 ... |                            ... |                          ... |         ... |       ... |                       ... |      ... |        ... |
>| 2017-08-25 04:41:00 |       theft-items-from-vehicle |     theft-from-motor-vehicle | -104.880586 | 39.645164 |             hampden-south |        1 |          0 |
>| 2017-09-13 04:17:00 |         theft-of-motor-vehicle |                   auto-theft | -105.028694 | 39.708288 |                  westwood |        1 |          0 |
>| 2017-09-13 02:21:00 |                 assault-simple | other-crimes-against-persons | -104.925733 | 39.654184 |          university-hills |        1 |          0 |
>| 2017-09-13 03:21:00 |      traffic-accident-dui-duid |             traffic-accident | -105.010711 | 39.757385 |                  highland |        0 |          1 |
>| 2017-09-13 02:15:00 |   traffic-accident-hit-and-run |             traffic-accident | -105.043950 | 39.787436 |                     regis |        0 |          1 |
>
>29078 rows × 7 columns

- 查看发生在某个时刻的犯罪记录

```python
crime.at_time('5:47')
```

><font color='red'>显示结果：</font>
>
>|                     |              OFFENSE_TYPE_ID |      OFFENSE_CATEGORY_ID |     GEO_LON |   GEO_LAT |          NEIGHBORHOOD_ID | IS_CRIME | IS_TRAFFIC |
>| ------------------: | ---------------------------: | -----------------------: | ----------: | --------: | -----------------------: | -------: | ---------: |
>|       REPORTED_DATE |                              |                          |             |           |                          |          |            |
>| 2013-11-26 05:47:00 |      criminal-mischief-other |          public-disorder | -104.991476 | 39.751535 |                      cbd |        1 |          0 |
>| 2017-04-09 05:47:00 |    criminal-mischief-mtr-veh |          public-disorder | -104.959394 | 39.678425 |               university |        1 |          0 |
>| 2017-02-19 05:47:00 |      criminal-mischief-other |          public-disorder | -104.986767 | 39.741336 |       north-capitol-hill |        1 |          0 |
>| 2017-02-16 05:47:00 |           aggravated-assault |       aggravated-assault | -104.934029 | 39.732320 |                     hale |        1 |          0 |
>| 2017-02-12 05:47:00 |          police-interference |         all-other-crimes | -104.976306 | 39.722644 |                    speer |        1 |          0 |
>|                 ... |                          ... |                      ... |         ... |       ... |                      ... |      ... |        ... |
>| 2013-09-10 05:47:00 |             traffic-accident |         traffic-accident | -104.986311 | 39.708426 |     washington-park-west |        0 |          1 |
>| 2013-03-14 05:47:00 |                  theft-other |                  larceny | -105.047861 | 39.727237 |               villa-park |        1 |          0 |
>| 2012-10-08 05:47:00 |     theft-items-from-vehicle | theft-from-motor-vehicle | -105.037308 | 39.768336 |            west-highland |        1 |          0 |
>| 2013-08-21 05:47:00 |     theft-items-from-vehicle | theft-from-motor-vehicle | -105.021310 | 39.758076 |           jefferson-park |        1 |          0 |
>| 2017-08-23 05:47:00 | traffic-accident-hit-and-run |         traffic-accident | -104.931056 | 39.702503 | washington-virginia-vale |        0 |          1 |
>
>118 rows × 7 columns

- 在按时间段选取数据时，可以将时间索引排序，排序之后再选取效率更高

```python
crime_sort = crime.sort_index()
%timeit crime.loc['2015-3-4':'2016-1-1']
```

><font color='red'>显示结果：</font>
>
>```shell
>15 ms ± 399 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
>```

```python
%timeit crime_sort.loc['2015-3-4':'2016-1-1']
```

><font color='red'>显示结果：</font>
>
>```shell
>1.59 ms ± 20 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
>```

- 计算每周的犯罪数量
  - 为了统计每周的犯罪数，需要按周分组
  - resample重采样，可以按照指定时间周期分组

```python
crime_sort.resample('W')
#size查看分组大小
weekly_crimes = crime_sort.resample('W').size()
weekly_crimes
```

><font color='red'>显示结果：</font>
>
>```shell
>REPORTED_DATE
>2012-01-08     877
>2012-01-15    1071
>2012-01-22     991
>2012-01-29     988
>2012-02-05     888
>              ... 
>2017-09-03    1956
>2017-09-10    1733
>2017-09-17    1976
>2017-09-24    1839
>2017-10-01    1059
>Freq: W-SUN, Length: 300, dtype: int64
>```

- 检验分组结果

```python
len(crime_sort.loc[:'2012-1-8'])
```

><font color='red'>显示结果：</font>
>
>```
>877
>```

```python
len(crime_sort.loc['2012-1-9':'2012-1-15'])
```

><font color='red'>显示结果：</font>
>
>```py
>1071
>```

- 也可以把周四作为每周的结束

```python
crime_sort.resample('W-THU').size()
```

><font color='red'>显示结果：</font>
>
>```shell
>REPORTED_DATE
>2012-01-05     462
>2012-01-12    1116
>2012-01-19     924
>2012-01-26    1061
>2012-02-02     926
>              ... 
>2017-09-07    1803
>2017-09-14    1866
>2017-09-21    1926
>2017-09-28    1720
>2017-10-05      28
>Freq: W-THU, Length: 301, dtype: int64
>```

- 将按周分组结果可视化

```python
weekly_crimes.plot(figsize=(16,4),title='丹佛犯罪情况')
```

><font color='red'>显示结果：</font>
>
>![img](data:image/png;base64,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)

- 分析每季度的犯罪和交通事故数据

```python
crime_quarterly = crime_sort.resample('Q')['IS_CRIME', 'IS_TRAFFIC'].sum()
crime_quarterly
```

><font color='red'>显示结果：</font>
>
>|               | IS_CRIME | IS_TRAFFIC |
>| ------------: | -------: | ---------: |
>| REPORTED_DATE |          |            |
>|    2012-03-31 |     7882 |       4726 |
>|    2012-06-30 |     9641 |       5255 |
>|    2012-09-30 |    10566 |       5003 |
>|    2012-12-31 |     9197 |       4802 |
>|    2013-03-31 |     8730 |       4442 |
>|    2013-06-30 |    12259 |       4510 |
>|    2013-09-30 |    15799 |       4942 |
>|    2013-12-31 |    13910 |       4968 |
>|    2014-03-31 |    14487 |       5021 |
>|    2014-06-30 |    15833 |       5225 |
>|    2014-09-30 |    17342 |       5734 |
>|    2014-12-31 |    15028 |       5783 |
>|    2015-03-31 |    14989 |       5380 |
>|    2015-06-30 |    16924 |       5825 |
>|    2015-09-30 |    17891 |       5988 |
>|    2015-12-31 |    16090 |       6117 |
>|    2016-03-31 |    16423 |       5590 |
>|    2016-06-30 |    17547 |       5861 |
>|    2016-09-30 |    17427 |       6199 |
>|    2016-12-31 |    15984 |       6094 |
>|    2017-03-31 |    16426 |       5587 |
>|    2017-06-30 |    17486 |       6148 |
>|    2017-09-30 |    17990 |       6101 |

- 所有日期都是该季度的最后一天，使用QS生成每季度的第一天

```python
crime_sort.resample('QS')['IS_CRIME', 'IS_TRAFFIC'].sum().head()
```

><font color='red'>显示结果：</font>
>
>|               | IS_CRIME | IS_TRAFFIC |
>| ------------: | -------: | ---------: |
>| REPORTED_DATE |          |            |
>|    2012-01-01 |     7882 |       4726 |
>|    2012-04-01 |     9641 |       5255 |
>|    2012-07-01 |    10566 |       5003 |
>|    2012-10-01 |     9197 |       4802 |
>|    2013-01-01 |     8730 |       4442 |

- 查看第二季度的数据，检验结果

```python
crime_sort.loc['2012-4-1':'2012-6-30', ['IS_CRIME', 'IS_TRAFFIC']].sum()
```

><font color='red'>显示结果：</font>
>
>```
>IS_CRIME      9641
>IS_TRAFFIC    5255
>dtype: int64
>```

- 结果可视化

```python
plot_kwargs = dict(figsize=(16,4), color=['black', 'blue'], title='丹佛犯罪和交通事故数据')
crime_quarterly.plot(**plot_kwargs)
```

><font color='red'>显示结果：</font>
>
>![img](data:image/png;base64,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24ULF2bL87kSEx4ebl9++WWbP39+W6xYMbtgwQKNEkiDqKgo+80339j777/f5s2b1wK2bdu2NjAw0IaFhbldnutOnz5tn3vuOVuwYEHr6+trBw8ebA8dOuR2WTnWwYMH7YQJE2LOIS5atKgdPny4/fnnn3P0v2+F1Uxwww03xDy+cOGC7dKli73lllts3759kzyY3nzzTdu2bVubP39+27ZtW7ty5Urbv39/27hxY9uyZUvbo0ePmG/3t2zZYqtUqRLz2ujl2rZta9u2bWuXLVuWaJu1zhC3W2+9Nea1ERERdsiQIbZ169a2X79+iZ6L4y37VUQkpdavXx8zLKpevXr2448/TtN/5u+995718/OzN9xwg/4WZnPBwcG2bdu2FrCAvfnmm+3MmTPtiRMn3C4t0+zdu9fecsstFrC33nqr3b9/v9slZQunTp2y06dPtzVr1rSALV68uH3yySdtcHCw26V5pePHj9vHH3/c+vn5WT8/P/voo49edYqFpM3Fixft+++/bzt27GiNMTH/lpcuXZptR3+kVo4Nq48/bm3bthk7Pf54ynZq3LA6Z84c+/LLL1trrR08ePA1T0iP+9r+/fvbDRs2WGutHTBggP3000+ttda+9NJLNk+ePDHD2OIul9hr40oYVpcuXWr79etnrbX2mWeesStWrLjqNfqAJiLZxfbt2+0999wTc9XWBQsW2IiIiHSt8/vvv7dlypSxRYsWtZ9//nkGVSpZJSoqyr755ps2f/78tmTJkvall16yBw8edLusLBMZGWnfeustW6RIEVuwYEE7Y8aMdP+byImioqLshg0b7AMPPGDz5ctnAXvTTTfZRYsW5ZpQkF6HDx+2Q4YMsb6+vrZAgQL22Wef1aiUNIiKirKbNm2yI0aMsP7+/jFX+x4/fnyOu3hXSiQXVrP5jXe8Q4UKFVi1ahW///478+fPp3nz5qleh7WWCxcu4OfnB8AXX3zBI488wueff57u+r744gvuvvtuAHr16kXp0qXTvU4Rkax25MgRBg0aRKNGjdiwYQMvvvgie/fuZdCgQfj6+qZr3TfddBObNm2iatWqdOrUiddff935Rle83tGjR7nzzjt59NFHad++Pdu3b+fZZ5+lcuXKbpeWZXx8fBg2bBi7du2iQ4cOPPXUU9x8883s2LHD7dK8wj///MObb75J/fr1adOmDR9/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cGFmz57Nl19+yYULF7jpppsYM2YMV65cycBqc7fw8HAWLlxInTp1+L//+z+stSxdupTdu3czaNCgFPeIS/r16NGDjRs3EhUVxc0338zq1avdLinFrLW8++67NGzYkK1bt7Jo0SKWLVumc5pFRCRD6dY1IpIua9eu5fbbb6dnz54EBgZm2P0T//nnH5544gnee+89GjVqFPNT0iYsLIyFCxfy8ssvc+jQIZo0acKYMWPo0qULPj763tJNx44do0uXLmzatInJkyfTo0cPfH19yZMnT8zPuI/j/vT19c3ye5aeOnWKoUOHsmrVKm655RYWLVpE5cqVs7QGERHJOXSfVRHJFIcPH6ZZs2aULVuWn376KVPOb1yzZg1Dhw7l7NmzjB8/nmeeeYY8efJk+HZyqtDQUObNm8crr7zCsWPHaNWqFWPGjOGuu+7K8pAjSbt06RKDBw8mMDAw1a+NDq3XCrfpaYv783//+x9nz55lypQpPPnkk/j6+mbCHhERkdxCYVVEMlxYWBht2rRh7969/PLLL5l6D8VTp07x8MMPs2zZMlq0aMF7771H7dq1M217OcG5c+eYNWsWr776KqdOnaJDhw6MGTOGdu3aKaR6KWstX3/9NadOnSIiIoLIyMh4P91qSzivSpUqzJkzh4YNG7q9y0REJAdILqyqe0JE0mTkyJEEBQWxZs2aTA2qAKVKleLDDz/kv//9Lw8//DBNmjThhRde4PHHH9cQ1gROnz7N66+/zhtvvMG5c+fo1KkTo0eP5qabbnK7NLkGYwy33Xab22WIiIh4DX3KE5FUe/vtt1mwYAGjR4+mc+fOWbbdXr16sWPHDjp27MhTTz1F+/bt+eOPP7Js+97s+PHjPPPMM1SuXJlJkyZx6623snnzZj755BMFVREREcmWFFZFJFU2bdrEyJEjueOOO5gwYUKWb/+6667jo48+4p133mHLli00bNiQuXPn4s2nNGSmP//8k8cee4yqVasyffp07rvvPnbs2MGKFSto2rSp2+WJiIiIpJnCqoik2MmTJ+nRowfly5dn6dKlrl1YxRjDwIED2bFjBy1btmT48OHceeedHDlyxJV63LB//36GDh3KDTfcwJw5c3jggQfYs2cPS5cupV69em6XJyIiIpJuCqsikiIRERH07t2bkydPsmLFCkqWLOl2SVSqVIkvv/ySWbNmsXHjRurXr8/ixYtzdC9rcHAwDz74IDVr1mTRokUMHTqUffv2MX/+fKpXr+52eSIiIiIZRmFVRFJk9OjRfPPNN8yZM8erhpf6+Pjw8MMPs3XrVurXr0+/fv3o1q0bJ06ccLu0DBMWFsbGjRvp2bMn9erVY+XKlTz55JMcOHCAmTNn6h6XIiIikiPp1jUick0rVqygR48ejBgxgtmzZ7tdTpIiIyN59dVXGTNmDEWKFGHOnDn06NHD7bJSJSIigl27dvHLL7/ETNu3byc8PJyiRYvy6KOP8sQTT1CqVCm3SxURERFJN91nVUTSLDg4mBYtWlC/fn3Wr1+Pn5+f2yVd065du+jXrx+bN2+mT58+zJw5kxIlSrhd1lWstezfvz9eMP3111+5ePEiAP7+/gQEBNC8eXOaN2/Orbfeir+/v8tVi4iIiGQchVURSZOQkBBatGjB2bNn+fXXX6lQoYLbJaVYeHg4L730EhMnTqR06dK8/fbb3H333a7W9Ndff8ULpkFBQZw9exaA/Pnz06RJE1q0aBETTqtXr677yIqIiEiOprAqIqlmraVHjx6sWbOGtWvX0rZtW7dLSpMtW7bQr18/duzYweDBg5kxYwZFixbN9O2eOXOGoKAgNm3aFBNOjx07BoCvry8NGjSICaXNmzenXr165M2bN9PrEhEREfEmyYXVPFldjIhkD1OnTmXlypVMnz492wZVgCZNmhAUFMT48eOZOnUqX331FQsXLqRDhw4Zto3Q0FB+/fXXeL2m+/fvj5lfs2ZNbr311phg2rhxYwoUKJBh2xcRERHJiVLUs2qMKQsst9a2SdD2ubW2ief5AqAu8Im1dnJq2pKinlURd6xdu5bbb7+dnj17EhgYiDHG7ZIyxE8//UT//v3Zu3cvI0eO5KWXXqJQoUKpWseVK1fYtm1bvGC6a9cuoqKiAKhYsWK8HtNmzZpRrFixTHg3IiIiItlfunpWjTHFgfeAhJ/opgEFPMt0A3ytta2MMe8YY2oADVLSZq39PR3vTUQy2KFDh+jVqxd16tRh/vz5OSaoArRs2ZItW7bw73//m9dff53PP/+c9957j5tuuinR5SMjI9mzZ09MKN20aRNbt27lypUrAJQsWZLmzZvTrVu3mHBatmzZrHxLIiIiIjlWSoYBRwK9gDXRDcaYDkAocNzT1A740PP4S6A10CSFbQqrIl4iLCyM7t27Ex4ezsqVKylcuLDbJWW4ggUL8tprr3HfffcxcOBA2rRpw6hRo5gwYQLHjh2L12O6efNmLly4AEDhwoVp1qwZjz32WEwwrVKlSo4K8yIiIiLe5Jph1VobAsR8IDPG+AFjga7Aas9ihYC/PI/PAE1T0RaPMWYoMBSgUqVKqXs3IpIuI0eOZPPmzaxZs4aaNWu6XU6mat++Pdu3b+fpp59m6tSpvPbaazE9pn5+fjRu3Jj+/fvHBNNatWrh6+vrctUiIiIiuUdaLrD0HDDbWvtPnB6FC3iGBAOFAZ9UtMVjrZ0HzAPnnNU01CciafD222+zYMECRo8eTefOnd0uJ0sUKVKEefPm0a1bNz7++GPq169P8+bNadiwYba4n6yIiIhITpbiW9cYY9ZZa9sZY74DojzNjYHlwHdAGWvtNGPMBGAPThC+Zpu19v2ktqkLLIlkjU2bNtGmTRvat2/PJ598oh5EEREREckSGXrrGmvtLXFWvM5a+5AxpiiwwRhTHrgLaAnYFLaJiItOnjxJjx49KF++PEuXLlVQFRERERGvcNUw3KRYa9sl1eY5r7Ud8BPQ3lp7LqVt6StfRNIjIiKC3r17c/LkSVasWEHJkiXdLklEREREBEjbOauJstaeJfZKv6lqExF3jB49mm+++YaFCxfStOlV1zsTEREREXFNintWRSRnWb58OVOnTmXEiBEMGDDA7XJEREREROJRWBXJhYKDgxk4cCAtW7bktddec7scEREREZGrKKyK5DIhISF07dqVggULsnz5ct2iRURERES8Uoadsyoi3s9ay8CBA9m3bx9r166lQoUKbpckIiIiIpIohVWRXGTq1KmsXLmS6dOn07ZtW7fLERERERFJkoYBi+QSa9eu5d///je9evXiySefdLscEREREZFkKayK5AKHDh2iV69e1KlTh/nz52OMcbskEREREZFkKayK5HBhYWF0796d8PBwVq5cSeHChd0uSURERETkmnTOqkgON3LkSDZv3syaNWuoWbOm2+WIiIiIiKSIelZFcrC3336bBQsWMHr0aDp37ux2OSIiIiIiKaawKpJDbdq0iZEjR3LHHXcwYcIEt8sREREREUkVhVWRHOjkyZP06NGD8uXLs3TpUnx9fd0uSUREREQkVXTOqkgOExERQe/evTl58iTff/89JUuWdLskEREREZFUU1gVyWFGjx7NN998w8KFC2natKnb5YiIiIiIpImGAYvkIMuXL2fq1KmMGDGCAQMGuF2OiIiIiEiaKayK5BDBwcEMHDiQli1b8tprr7ldjoiIiIhIuiisiuQAISEhdO3alYIFC7J8+XL8/PzcLklEREREJF10zqpINmetZeDAgezbt4+1a9dSoUIFt0sSEREREUk3hVWRbG7q1KmsXLmS6dOn07ZtW7fLERERERHJEBoGLJKNff311/z73/+mV69ePPnkk26XIyIiIiKSYRRWRbKpQ4cO0bt3b+rUqcP8+fMxxrhdkoiIiIhIhklRWDXGlDXGbPA89jfGfGaM+dIYs8oY4+dpX2CM+dEYMybO61LUJiKpc+nSJbp160Z4eDirVq2icOHCbpckIiIiIpKhrhlWjTHFgfeAQp6mB4AZ1trbgePAncaYboCvtbYVUM0YUyOlbZnxpkRyMmstDz/8ML/++itLliyhRg39MxIRERGRnCclPauRQC8gBMBaO9ta+5VnXmngb6Ad8KGn7UugdSra4jHGDDXGBBljgk6ePJm6dyOSC8ydO5d3332X//znP9x7771ulyMiIiIikimuGVattSHW2nMJ240xrYDi1tqfcHpd//LMOgOUTUVbwu3Ns9YGWGsDSpcuncq3I5Kz/fTTTzz22GPcddddjBs3zu1yREREREQyTZpuXWOMKQG8CXT3NF0ACngeF8YJwSltE5EUOHHiBN27d6dixYosWbIEHx/98xERERGRnCvVn3Y9F1RaBjxvrT3kad5M7JDeRsDBVLSJyDWEh4dz//33c/bsWVauXEmJEiXcLklEREREJFOlpWd1MNAUGG2MGQ3MAVYDG4wx5YG7gJaATWGbiFzDM888w3fffceSJUto1KiR2+WIiIiIiGQ6Y63NmBU5Vw2+DfjOWns8NW1JCQgIsEFBQRlSn0h2FRgYyP/93//x2GOP8frrr7tdjoiIiIhIhjHGbLbWBiQ6L6PCamZQWJXcbtu2bbRs2ZKAgADWrl1L3rx53S5JRERERCTDJBdWdYUWES919uxZunXrRrFixfjwww8VVEVEREQkV0nT1YBFJHNFRUXRt29fDh8+zPr16ylXrpzbJYmIiIiIZCmFVREvNGnSJD799FNmzZpFq1at3C5HRERERCTLaRiwiJf53//+x/jx4+nfvz8jRoxwuxwREREREVcorIp4kX379tG3b1+aNGnCnDlzMMa4XZKIiIiIiCsUVkW8RGhoKF27dsXX15eVK1dSoEABt0sSEREREXGNzlkV8QLWWh566CF27drF559/TpUqVdwuSURERETEVQqrIl7g9ddf54MPPuCFF17gtttuc7scERERERHXaRiwiMvWr1/PqFGj6Nq1K88995zb5YiIiIiIeAWFVREXHTlyhPvvv5/q1avz7rvv6oJKIiIiIiIeGgYs4pLLly/To0cPLl68yLp16yhatKjbJYmIiIiIeA2FVRGXPPHEE/z8888sX76cOnXquF2OiIiIiIhX0TDgHC4qKsrtEiQR77zzDm+99RbPPvss3bt3d7scERERERGvo7CaA+3Zs4fJkyfTsGFDChQowIgRIzh06JDbZYlHUFAQDz/8MB07dmTy5MlulyMiIiIi4pUUVnOI3bt3M2nSJBo2bEjt2rUZO3YsRYsWpVevXixYsIDq1aszZMgQ/vjjD7dLzdVOnTpF9+7dKVu2LIGBgeTJo5H4IiIiIiKJUVjNxoKDg5k4cSINGjSgTp06jBs3Dn9/f15//XWOHDnCxo0bWbRoEfv372f48OEsXryYmjVrMnDgQH7//Xe3y891IiIi6N27NydOnGDlypWUKlXK7ZJERERERLyWsda6XUOSAgICbFBQkNtleJXg4GCWLVvGsmXL2LFjB8YYWrduTc+ePenevTvly5dP8rVHjx7llVde4a233uLKlSv83//9H6NHj6Z27dpZ+A5yr+eee46XX36Zd955h4EDB7pdjoiIiIiI64wxm621AYnOU1j1frt27YoJqDt37kxVQE3M8ePHmT59OrNnz+bSpUv06tWLMWPGUK9evUx6B7JixQp69OjB8OHDmTNnjtvliIiIiIh4BYXVbCixgNqmTRt69uxJt27dUh1QE3Py5ElmzJjBzJkzuXDhAj169GDs2LE0bNgwA96BRAsODqZFixbUr1+fdevWkS9fPrdLEhERERHxCgqr2cTOnTtjAuquXbviBdTu3btz3XXXZcp2T58+zWuvvcYbb7xBSEgIXbp0YezYsTRt2jRTtpebhISE0KJFC86ePcuvv/5KhQoV3C5JRERERMRrJBdWU3SBJWNMWWPMhjjPFxhjfjTGjMmIttxs586djBs3jrp161K/fn0mTpxI6dKlmTlzJn/99Rfr169n5MiRmRZUAUqWLMmkSZM4ePAg48ePZ926dTRr1ox7772XTZs2Zdp2c7qoqCj69+/Pvn37+PDDDxVURURERERS4Zph1RhTHHgPKOR53g3wtda2AqoZY2qkpy2z3pi3stayY8eOeAF10qRJlClThlmzZnH06FHWrVvHI488kqkBNTHFixdn3LhxHDx4kMmTJ/PDDz9w4403ctddd/Hjjz9maS05wcsvv8zq1auZNm0abdu2dbscEREREZFs5ZrDgI0xRQEDrLHWtjPGvAF8bq391BjTGygANElrm7V2YYLtDQWGAlSqVKnZoUOHMvQNu8Fay86dO/nwww9ZtmwZu3fvxsfHh1tuuSXmHNRy5cq5XeZVzp8/z+zZs5k2bRqnTp2iY8eO/Oc//6FNmzZul+b1vvzyS+6880569+7N0qVLMca4XZKIiIiIiNdJ1zBga22ItfZcnKZCwF+ex2eAsulsS7i9edbaAGttQOnSpa9Vntey1rJ9+3b+85//ULduXRo0aMCUKVO47rrrmD17NkePHuXbb7/l4Ycf9sqgClCkSBGeffZZDh48yLRp09i2bRu33HIL7du359tvv8Wbz3d208GDB+nTpw/169fn7bffVlAVEREREUmDFJ2zmsAFnF5SgMKedaSnLcew1rJt2zbGjh1LnTp1aNiw4VUB9ZtvvmHEiBGULXtVTvdahQoV4umnn+bAgQO89tpr7Nmzhw4dOnDLLbfw1VdfKbTGcenSJbp160ZkZCQrV66kUKFCbpckIiIiIpItpSUsbgZaex43Ag6msy1biw6oY8aMoXbt2jRq1IgXXniBChUqMGfOnGwbUBNTsGBBHn/8cf744w9mzpzJwYMHuf3227npppv47LPPcn1otdYyfPhwtmzZwtKlS6levbrbJYmIiIiIZFt50vCa1cAGY0x54C6gJWDT0ZYtXbp0iRkzZrBo0SL27t2Lj48P7dq148knn6Rbt26UKVPG7RIzTf78+XnkkUd46KGHePfdd3nhhRfo1KkTAQEB/Oc//+Gee+7JlUNf58yZw6JFixg/fjx333232+WIiIiIiGRrabrPqucKwbcB31lrj6e3LSneep/Vr7/+muHDh7N//346dOjA/fffT9euXXN0QE3OlStXWLx4MVOmTOHAgQM0adKEsWPHct999+Hjk6NGeifphx9+oF27dtx+++189NFHueZ9i4iIiIikR3IXWEpTWM0q3hZWT506xdNPP82iRYuoUaMGc+fOpX379m6X5TXCw8NZunQpU6ZMYd++fTRo0ICxY8fSvXv3HB3ejh8/TtOmTSlYsCC//PILxYsXd7skEREREZFsIV1XAxbnXMRFixZRu3Zt3n//fcaMGcO2bdsUVBPImzcvAwYMIDg4mCVLlhAeHs79999PgwYNCAwMJDIy0u0SM1x4eDg9e/bk3LlzrFy5UkFVRERERCSDKKxew759+7jtttvo378/NWvWZMuWLUyaNIn8+fO7XZrXypMnDw888AA7duzggw8+wBjD//3f/1G3bl0WL15MRESE2yVmmFGjRrFx40bmz59Pw4YN3S5HRERERCTHUFhNQnh4OC+99BINGjTgl19+Yfbs2WzcuJH69eu7XVq24evrS69evdi2bRvLly8nf/789OvXj9q1a7Nw4ULCw8PdLjFdli5dyhtvvMETTzxBnz593C5HRERERCRHUVhNxM8//0xAQADPP/88nTp1YteuXYwYMSJHn3eZmXx8fOjevTtbtmxh9erV+Pv7M2jQIGrWrMnbb79NWFiY2yWm2tatWxkyZAi33HILU6dOdbscEREREZEcRxdYiuP8+fOMHj2amTNnUr58eWbNmsV9992XZdvPLay1fPrpp0yYMIFffvkFX19fatSoQb169eJNNWrUwM/Pz+1yr3LmzBkCAgK4fPkyv/76a7a/f66IiIiIiFuSu8BSWu6zmiN99NFHPPLII/z111+MHDmSyZMnU7RoUbfLypGMMdx999106tSJtWvXsm7dOnbu3Mm2bdtYtWoVUVFRgHPua82aNa8KsdWrVydv3ryu1B4VFUXfvn05cuQI3333nYKqiIiIiEgmyfVh9ejRozz22GOsWLGCBg0asHz5cm688Ua3y8oVjDF07NiRjh07xrRdunSJPXv2sHPnzpjp119/Zfny5USPAsibNy+1atW6KsTecMMN5MmTuYf0hAkT+Oyzz3jrrbdo2bJlpm5LRERERCQ3y7XDgKOiopg3bx7PPvssV65cYdy4cTz99NOu9dhJ8i5evMju3bvjhdidO3dy4MCBmGX8/PyoXbv2VSG2WrVq+Pr6pruGjz/+mM6dOzNw4EAWLFiAMSbd6xQRERERyc2SGwacK8Pqzp07GTp0KD/88AO33norb731FtWrV8/w7UjmCw0NJTg4+KoQe+jQoZhl8ufPHxNi69atGxNiq1atmuIQ+/vvvxMQEECNGjXYsGEDBQoUyKy3JCIiIiKSayiseoSFhTFlyhRefvllihYtyowZM3jwwQfVQ5YDXbhwgV27dl0VYv/888+YZQoUKJBoT2yVKlXiXfn5woULtGzZkuPHj7N582YqV67sxlsSEREREclxdIElYN26dQwdOpTff/+dfv36MX36dEqVKuV2WZJJChcuTIsWLWjRokW89pCQkHghdteuXaxbt44lS5bELFOwYEHq1KkTE16///57goOD+eKLLxRURURERESySI4Pq2fOnOFf//oX77zzDtWqVeOrr76Kd0EfyV2KFi1Ky5Ytr7o40rlz567qif36669ZtGgRAC+//LKOGxERERGRLJRjw6q1lg8++IAnnniC06dP89xzzzF27FgKFizodmnihfz9/WnVqhWtWrWK13727FnOnj1LtWrVXKpMRERERCR3ypFh9cCBAzz88MN8/vnntGjRgi+//JJGjRq5XZZkQ8WLF6d48eJulyEiIiIikuv4XHuR7CMiIoJp06ZRv359Nm7cyBtvvMEPP/ygoCoiIiIiIpLN5Jie1aCgIIYOHcqWLVvo3LkzM2fOpGLFim6XJSIiIiIiImmQ7XtWL1y4wFNPPcWNN97I8ePHWbFiBatXr1ZQFRERERERycaydc/qp59+yogRIzh8+DAjRozgxRdfxN/f3+2yREREREREJJ2yZc/q8ePH6d27N3fffTeFCxdm48aNzJ49W0FVREREREQkh8hWYTUqKor58+dTp04dVq1axaRJk9iyZQs333yz26WJiIiIiIhIBkp1WDXGFDfGfGqMCTLGzPW0LTDG/GiMGRNnuRS1pdTu3btp164dQ4YMoVGjRmzbto0xY8bg5+eX2lWJiIiIiIiIl0tLz+qDwFJrbQBQxBjzDOBrrW0FVDPG1DDGdEtJW0o2dvnyZSZMmECjRo3YsWMHCxYs4Ntvv6VWrVppKF1ERERERESyg7RcYOk0UN8YUwyoCJwDPvTM+xJoDTRJYdvvCVdujBkKDAUoW7YsjRs3Zvfu3fTp04dXX32VsmXLpqFkERERERERyU7SElY3AncDjwHBgB/wl2feGaApUCiFbVex1s4D5gEYY2yBAgX47LPPuPPOO9NQqoiIiIiIiGRHaQmr44Dh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)

- 分析工作日的犯罪情况：可以通过Timestamp的dt属性得到周几，然后统计

```python
crime = pd.read_csv('data/crime.csv', parse_dates=['REPORTED_DATE'])
wd_counts = crime['REPORTED_DATE'].dt.weekday.value_counts()
wd_counts
```

><font color='red'>显示结果：</font>
>
>```shell
>0    70024
>4    69621
>2    69538
>3    69287
>1    68394
>5    58834
>6    55213
>Name: REPORTED_DATE, dtype: int64
>```

```python
title = '丹佛犯罪和交通事故按周分析'
wd_counts.plot(kind='barh', title=title)
```

><font color='red'>显示结果：</font>
>
>![img](data:image/png;base64,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)

## 小结

- Pandas中，datetime64用来表示时间序列类型
- 时间序列类型的数据可以作为行索引，对应的数据类型是DatetimeIndex类型
- datetime64类型可以做差，返回的是Timedelta类型
- 转换成时间序列类型后，可以按照时间的特点对数据进行处理
  - 提取日期的各个部分（月，日，星期...)
  - 进行日期运算
  - 按照日期范围取值