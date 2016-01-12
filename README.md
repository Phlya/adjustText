# adjustText

```python
mtcars = pd.read_csv('../tmp/mtcars.csv')
labels = mtcars['Unnamed: 0']
f = plt.figure()
r = f.canvas.get_renderer()
xs, ys = mtcars['wt'], mtcars['mpg']
paths = plt.scatter(xs, ys)
texts = []
for x, y, s in zip(xs, ys, labels):
    texts.append(plt.text(x, y, s, bbox={'pad':0.0, 'alpha':0}, size=7))
plt.grid()

#plt.show()
```
![alt tag](https://raw.github.com/Phlya/adjustText/master/mtcars_before.png)
```python
adjust_text(xs, ys, texts, arrowprops=dict(arrowstyle='->', color='red'), bbox={'pad':0, 'alpha':0}, size=7)
plt.show()
```
![alt tag](https://raw.github.com/Phlya/adjustText/master/mtcars_after.png)
