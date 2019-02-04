from bokeh.plotting import figure, output_file, show
from bokeh.models import CategoricalColorMapper, ColumnDataSource
from bokeh.palettes import Category10

import umap
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()

data = iris.data
labels = iris.target
names = iris.target_names

shuffle_idxs = [x for x in np.arange(len(data))]
np.random.shuffle(shuffle_idxs)

data = data[shuffle_idxs]
labels = labels[shuffle_idxs]

split = 0.9
split = int(len(data)*split)
d1, d2 = data[:split], data[split:]
l1, l2 = labels[:split], labels[split:]

u = umap.UMAP(n_neighbors=50, # critical!
                      learning_rate=0.5,
                      init="random",
                      min_dist=0.001
                    )
embedding = u.fit_transform(d1)
oo_sample = u.transform(d2)

output_file("iris_unlabeled.html")

source1 = ColumnDataSource(dict(
    x=[e[0] for e in embedding],
    y=[e[1] for e in embedding],
    label=[names[d] for d in l1]
))

source2 = ColumnDataSource(dict(
    x=[e[0] for e in oo_sample],
    y=[e[1] for e in oo_sample],
    label=[names[d] for d in l2]
))

cmap = CategoricalColorMapper(factors=names, palette=Category10[10])

p = figure(title="test umap")
p.circle(x='x',
         y='y',
         source=source1,
         color={"field": 'label', "transform": cmap},
         legend='label')

p.square(x='x',
          y='y',
          source=source2,
          color={"field": 'label', "transform": cmap},
          legend='label')

show(p)
