# adjustText

Inspired by **ggrepel** package for R/ggplot2 (https://github.com/slowkow/ggrepel)

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

The algorithm is highly configurable to adjust it to your particular case. Getting figures convenient to work with is usually very easy.
