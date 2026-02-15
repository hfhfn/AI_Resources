# anaconda中启动jupyter notebook自动打开浏览器以及默认目录路径的设置

## 1.自动打开浏览器设置
在cmd命令符窗口输入

```
jupyter notebook --generate-config
```
在窗口中会显示jupyter_notebook_config.py这个文件的位置，一般的路径为C:\Users\自己的用户名\.jupyter\jupyter_notebook_config.py
### 修改jupyter_notebook_config.py文件，在最后面加上

```python
import webbrowser

webbrowser.register("chrome",None,webbrowser.GenericBrowser(r"C:\ProgramFiles(x86)\Google\Chrome\Application\chrome.exe"))

c.NotebookApp.browser = 'chrome'   
```
谷歌浏览器版本低的可能无法打开,可以设置其它的浏览器,只需要修改代码中引号部分,换成相应的浏览器即可
## 2.修改jupyter notebook默认路径
首先打开jupyter_notebook_config.py文件,进行修改,

```python
## The directory to use for notebooks and kernels.
c.NotebookApp.notebook_dir = '' # 将此行代码注释打开
```

```python
## The directory to use for notebooks and kernels.
c.NotebookApp.notebook_dir = 'D:/myprograms/anacond' # 输入我们想要默认打开的路径
```
