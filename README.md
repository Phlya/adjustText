# adjustText

Inspired by **ggrepel** package for R/ggplot2 (https://github.com/slowkow/ggrepel)

The idea is that often when we want to label multiple points on a graph the text will start heavily overlapping with both other labels and data points. This can be a major problem requiring manual solution. However this can be largely automatized by smart placing of the labels (difficult) or iterative adjustment of their positions to minimize overlaps (relatively easy). This library (well... script) implements the latter option to help with matplotlib graphs. Usage is very straightforward with usually pretty good results with no tweaking (most important is to just make text slightly smaller than default and maybe the figure a little larger). However the algorithm itself is highly configurable, but there is no documentation now, just see the docstring to the main function `adjust_text`.

```python
mtcars = pd.read_csv('mtcars.csv')
labels = mtcars['Car']
xs, ys = mtcars['wt'], mtcars['mpg']
plt.scatter(xs, ys, s=15, c='r', edgecolors=(1,1,1,0))
texts = []
for x, y, s in zip(xs, ys, labels):
    texts.append(plt.text(x, y, s, bbox={'pad':0, 'alpha':0}, size=7))

#plt.show()
```
![alt tag](https://raw.github.com/Phlya/adjustText/master/examples/mtcars_before.png)
```python
adjust_text(xs, ys, texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5),
            bbox={'pad':0, 'alpha':0}, size=7)
plt.show()
```
![alt tag](https://raw.github.com/Phlya/adjustText/master/examples/mtcars_after.png)

The process can be illustrated by the following animation:
![alt tag](https://raw.github.com/Phlya/adjustText/master/examples/animation.gif)
