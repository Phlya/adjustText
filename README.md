# adjustText

Inspired by **ggrepel** package for R/ggplot2 (https://github.com/slowkow/ggrepel)

See usage examples [here].

The idea is that often when we want to label multiple points on a graph the text will start heavily overlapping with both other labels and data points. This can be a major problem requiring manual solution. However this can be largely automatized by smart placing of the labels (difficult) or iterative adjustment of their positions to minimize overlaps (relatively easy). This library (well... script) implements the latter option to help with matplotlib graphs. Usage is very straightforward with usually pretty good results with no tweaking (most important is to just make text slightly smaller than default and maybe the figure a little larger). However the algorithm itself is highly configurable, but there is no documentation now, just see the docstring to the main function `adjust_text`.

Should be installable from pipy!
```
pip install adjustText
```

[here]: https://github.com/Phlya/adjustText/blob/master/examples/Examples.ipynb
