> 错误代码: Tinymind上运行word2vec_basic.py 
>
> 抛出异常:
>
> ```
> Most common words (+UNK) Traceback (most recent call last):
  File "/tinysrc/word2vec_basic.py", line 111, in 
    print('Most common words (+UNK)', count[:5]) 
UnicodeEncodeError: 'ascii' codec can't encode character '\u3002' in position 18: ordinal not in range(128)
> ```
>
> 原因分析:
>
> python解释器或是print为编码方式为ascii, 要更换为utf-8
>
> 代码修改: 
>
> ```python
> import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
> ```
>
> 

> 错误代码: win10 本地运行word2vec_basic.py 
>
> 抛出异常: 
>
> ```
> matplotlib.pyplot保存的图中中文无法显示,全部为口
> ```
>
> 原因分析:
>
> matplotlib.pyplot默认字体无法显示中文
>
> 代码修改: 
>
> ```python
> plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体为黑体
> """西方国家字母体系分为两类：serif 以及sans-serif。
> serif 是有衬线字体，意思是在字的笔画开始、结束的地方有额外的装饰，而且笔画的粗细会有所不同。相反的，sans-serif 就没有这些额外的装饰，而且笔画的粗细差不多。
> 简而言之就是:sans-serif为等宽字体,serif为普通字体
> """
> ```
>
> 

> 错误代码: ubuntu16 本地运行word2vec_basic.py 
>
> 抛出异常: 
>
> ```
> matplotlib.pyplot保存的图中中文无法显示,全部为口
> ```
>
> 原因分析:
>
> matplotlib.pyplot默认字体无法显示中文
>
> 代码修改: 
>
> ```python
> #plt.rcParams['font.sans-serif'] = ['SimHei'] 这句代码没用, ubuntu16中没有对应的字体
> zhfont = mpl.font_manager.FontProperties(fname='./heiti.ttf')
> plt.annotate('中文字体测试',  xy=(x, y), xytext=(5, 2),
>              textcoords='offset points', ha='right',
>              va='bottom',  fontproperties=zhfont)  #用fontproperties指定自己下载好的中文字体
> """
> 没有办法, 自己下了一个中文字体, 保存在本地, 并在用到中文字段的地方, 指定已下载好的字体
> """
> ```
>
> 
> 错误代码: Tinymind上运行word2vec_basic.py 
>
> 抛出异常:
>
> ```
> Traceback (most recent call last):
  File "/tinysrc/word2vec_basic.py", line 367, in 
    plot_with_labels(low_dim_embs, labels, os.path.join(output_dir, 'tsne.png'))
  File "/tinysrc/word2vec_basic.py", line 339, in plot_with_labels
    plt.figure(figsize=(18, 18))  # in inches
  File "/opt/conda/lib/python3.6/site-packages/matplotlib/pyplot.py", line 534, in figure
    **kwargs)
  File "/opt/conda/lib/python3.6/site-packages/matplotlib/backend_bases.py", line 170, in new_figure_manager
    return cls.new_figure_manager_given_figure(num, fig)
  File "/opt/conda/lib/python3.6/site-packages/matplotlib/backend_bases.py", line 176, in new_figure_manager_given_figure
    canvas = cls.FigureCanvas(figure)
  File "/opt/conda/lib/python3.6/site-packages/matplotlib/backends/backend_qt5agg.py", line 35, in __init__
    super(FigureCanvasQTAggBase, self).__init__(figure=figure)
  File "/opt/conda/lib/python3.6/site-packages/matplotlib/backends/backend_qt5.py", line 235, in __init__
    _create_qApp()
  File "/opt/conda/lib/python3.6/site-packages/matplotlib/backends/backend_qt5.py", line 122, in _create_qApp
    raise RuntimeError('Invalid DISPLAY variable')
  RuntimeError: Invalid DISPLAY variable
> ```
>
> 原因分析:
>
> 在Windows下使用matplotlib绘图可以，但是在ssh远程绘图的时候报错了
  matplotlib的默认backend是TkAgg，而FltAgg、GTK、GTKCairo、TkAgg、Wx和WxAgg这几个backend都要求有GUI图形界面，所以在ssh操作的时候会报错。
>
> 代码修改: 
>
> ```python
> plt.switch_backend('agg')  
> ```
>
> 

> 错误代码: 本地运行train.py 
>
> 抛出异常:
>
> ```
> python TypeError: Expected int32, got list containing Tensors of type '_Message' instead.
> ```
>
> 原因分析:
>
> tensorflow版本的问题, tensorflow1.0及以后api定义：(数字在后，tensors在前),
>
> 代码修改: 
>
> ```python
> tf.concat(seq_output, 1)
> ```
>
> 






