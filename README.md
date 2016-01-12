# adjustText

```
mtcars = pd.read_csv('../tmp/mtcars.csv')
mtcars.index=mtcars['Unnamed: 0']
f = plt.figure()
r = f.canvas.get_renderer()
xs, ys = mtcars['wt'], mtcars['mpg']
paths = plt.scatter(xs, ys)
texts = []
for x, y, s in zip(xs, ys, mtcars.index):
    texts.append(plt.text(x, y, s, bbox={'pad':0.0, 'alpha':0}))
plt.grid()

#plt.show()
```
<a target="_blank" href="http://itmages.ru/image/view/3385922/88429ef0"><img src="http://storage8.static.itmages.ru/i/16/0110/s_1452393957_2087440_88429ef063.png" /></a>
```
iteratively_repel_text_annotaions(xs, ys, texts, arrowprops=dict(arrowstyle='->', color='red'), bbox={'pad':0, 'alpha':0})
plt.show()
```

<a target="_blank" href="http://itmages.ru/image/view/3385923/c4636a1f"><img src="http://storage9.static.itmages.ru/i/16/0110/s_1452394029_2931634_c4636a1f1d.png" /></a>
