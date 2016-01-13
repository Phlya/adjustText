#Examples
#####Some usage examples and ways to make the figure better than what default options yield.

######A quite simple example from http://stackoverflow.com/q/19073683/1304161, but we'll make quite a nice figure in the end.

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
f = interpolate.interp1d(eucs, covers)
x = np.arange(min(eucs), max(eucs), 0.0005)
y = f(x)
```

However, the function assumes that if you use the number of the text in `texts` and get the respective `x` and `y` values, that that's where the text "point" to. To circumvent that we do this:
```python
x = eucs+list(x)
y = covers+list(y)
```
We want to avoid moving the labels along the x-axis, because, well, why not do it for illustrative purposes. For that we use the parameter `move_only={'points':'y', 'text':'y'}`. If we want to move them along x axis only in the case that they are overlapping with text, use `move_only={'points':'y', 'text':'xy'}`. We also want absolutely no overlaps (or run for the whole 100 default iteration), so set `precision=0`. All together:
```python
np.random.seed(1543)
adjust_text(x, y, texts, only_move={'points':'y', 'text':'y'}, precision=0, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
plt.show()
```
![alt tag](https://raw.github.com/Phlya/adjustText/master/examples/test_lines_fancy.png)

And this is now great, isn't it? Not perfect, but close to it.
However remember, that there is some randomness in the algorithm, and the exact output slightly varies between runs. That's why I did np.random.seed(1543) in the last code snippet, so that this way the output would stay the same.

