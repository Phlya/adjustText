#Examples
#####Some usage examples and ways to make the figure better than what default options yield.

######A very simple example from http://stackoverflow.com/q/19073683/1304161, but we'll make quite a nice figure in the end.

```python
together = [(0, 1.0, 0.4), (25, 1.0127692669427917, 0.41), (50, 1.016404709797609, 0.41), (75, 1.1043426359673716, 0.42), (100, 1.1610446924342996, 0.44), (125, 1.1685687930691457, 0.43), (150, 1.3486407784550272, 0.45), (250, 1.4013999168008104, 0.45)]
together.sort()

text = [x for (x,y,z) in together]
eucs = [y for (x,y,z) in together]
covers = [z for (x,y,z) in together]

p1 = plt.plot(eucs,covers,color="black", alpha=0.5)
texts = []
for x, y, s in zip(eucs, covers, text):
    texts.append(plt.text(x, y, s))

plt.xlabel("Proportional Euclidean Distance")
plt.ylabel("Percentage Timewindows Attended")
plt.title("Test plot")
#plt.show()
```
![alt tag](https://raw.github.com/Phlya/adjustText/master/examples/test_lines.png)

First let's just apply the function with default options (this is a simple case, so we'll just set `precision` to 0)
```python
#All of the above code
adjust_text(eucs, covers, texts, precision=0, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
#plt.show()
```
![alt tag](https://raw.github.com/Phlya/adjustText/master/examples/test_lines_default.png)

This is better, but you see that all the labels now moved and sometimes overlap the lines even more. This can be avoided, if we create "virtual" points along the line and supply them to the function to avoid them.
```python
import numpy as np
from scipy import interpolate
f = interpolate.interp1d(eucs, covers)
x = np.arange(min(eucs), max(eucs), 0.0005)
y = f(x)
```
Our new points cover the curves densely enough that we don't need to pass the locations of our original points to the function - the texts will be repelled from the points automatically, because they will be repelled from the lines in general. So we'll just use our new `x` and `y` variables as point coordinates. Just to mention, a caveat of this method is that providing too many points will repel the texts too far away (and you might want that with more complicated curves). If you have to put them very densely, try setting `force_points` to a small value, but it might be difficult to find a balance.

We want to avoid moving the labels along the x-axis, because, well, why not do it for illustrative purposes. For that we use the parameter `move_only={'points':'y', 'text':'y'}`. If we want to move them along x axis only in the case that they are overlapping with text, use `move_only={'points':'y', 'text':'xy'}`. All together:
```python
np.random.seed(1543)
adjust_text(x, y, texts, only_move={'points':'y', 'text':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
plt.show()
```
![alt tag](https://raw.github.com/Phlya/adjustText/master/examples/test_lines_fancy.png)

And this is now great, isn't it?
However remember, that there is some randomness in the algorithm, and the exact output slightly varies between runs. That's why I did np.random.seed(1543) in the last code snippet, so that this way the output would stay the same.

######A real-world example of a volcano plot
Volcano plots are frequently used in bioinformatics and most frequently show fold change of level of gene expression together with statistical significance of the change. This example is taken from http://www.gettinggeneticsdone.com/2016/01/repel-overlapping-text-labels-in-ggplot2.html A test of ggrepel package for R/ggplot2. Now we'll make a similar graph with highlighting and labelling of genes that changed there expression statistically significantly, and apply this library.

```python
#Load the data
import pandas as pd
data = pd.read_csv('volcano_data.csv')
#Now make the plot
plt.figure(figsize=(7, 10))
xns, yns = data['log2FoldChange'][data['padj']>=0.05], -np.log10(data['pvalue'][data['padj']>=0.05])
plt.scatter(xns, yns, c='grey', edgecolor=(1,1,1,0), label='Not Sig')
xs, ys = data['log2FoldChange'][data['padj']<0.05], -np.log10(data['pvalue'][data['padj']<0.05])
plt.scatter(xs, ys, c='r', edgecolor=(1,1,1,0), label='FDR<5%')
texts = []
for x, y, l in zip(xs, ys, data['Gene'][data['padj']<0.05]):
    texts.append(plt.text(x, y, l, size=8, bbox={'pad':0, 'alpha':0}))
plt.legend()
plt.xlabel('$log_2(Fold Change)$')
plt.ylabel('$-log_{10}(pvalue)$')
#plt.show()
```
That's what we get:

![alt tag](https://raw.github.com/Phlya/adjustText/master/examples/volcano_before.png)

We can't read some names of the genes on the left, the ones that are downregulated - fold change of their expression level is below 1 (or, equivalently, log2 of fold change is negative). Let's try to  fix it! We also don't want the texts to overlap with grey (non-significant) points, so we'll supply coordincates of all the points to the function.
```python
np.random.seed(2016)
adjust_text(data['log2FoldChange'], -np.log10(data['pvalue']), texts,
            arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
plt.show()
```
![alt tag](https://raw.github.com/Phlya/adjustText/master/examples/volcano_after.png)

Now that's better!
