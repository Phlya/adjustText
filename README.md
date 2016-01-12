# adjustText

Inspired by **ggrepel** package for R/ggplot2 (https://github.com/slowkow/ggrepel)

```python
mtcars = pd.read_csv('mtcars.csv')
labels = mtcars['Car']
f = plt.figure()
r = f.canvas.get_renderer()
xs, ys = mtcars['wt'], mtcars['mpg']
paths = plt.scatter(xs, ys)
texts = []
for x, y, s in zip(xs, ys, labels):
    texts.append(plt.text(x, y, s, bbox={'pad':0.0, 'alpha':0}, size=7))

#plt.show()
```
![alt tag](https://raw.github.com/Phlya/adjustText/master/examples/mtcars_before.png)
```python
adjust_text(xs, ys, texts, arrowprops=dict(arrowstyle='-', color='k'), bbox={'pad':0, 'alpha':0}, size=7)
plt.show()
```
![alt tag](https://raw.github.com/Phlya/adjustText/master/examples/mtcars_after.png)

The algorithm is highly configurable to adjust it to your particular case. Getting figures convenient to work with is very easy, to get high-quality figures for e.g. publishing can be more complicated.
