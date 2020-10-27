from sklearn import datasets
import matplotlib.pyplot as plt


#Load the digits dataset
digits = datasets.load_digits()


#Display the first digit
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

# Now want to configure kmapper such that it shows helpful information for members of a node in the html-visualization
import io
import base64
import numpy as np
import imageio
from PIL import Image

tooltip_s = []
for image_data in digits.data:
    output = io.BytesIO()
    img = Image.fromarray(image_data.reshape((8, 8))) # Data was a flat row of 64 "pixels".
    imageio.imwrite(output, img, format="PNG")
    contents = output.getvalue()
    img_encoded = base64.b64encode(contents)
    img_tag = """<img src="data:image/png;base64,{}">""".format(img_encoded.decode('utf = 8'))
    tooltip_s.append(img_tag)
    output.close()

tooltip_s = np.array(tooltip_s) # need to make sure to feed it as a NumPy array, not a list

import sklearn
import kmapper as km
mapper = km.KeplerMapper(verbose=2)

# Fit and transform data
projected_data = mapper.fit_transform(digits.data, projection=sklearn.manifold.TSNE())

# Create the graph (we cluster on the projected data and suffer projection loss)
graph = mapper.map(projected_data,
                   clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=15),
                   cover=km.Cover(35, 0.4))

# Create the visualizations (increased the graph_gravity for a tighter graph-look.)
print("Output graph examples to html" )
# Tooltips with image data for every cluster member
mapper.visualize(graph,
                 title="Handwritten digits Mapper",
                 path_html="digits_custom_tooltips.html",
                 color_function=digits.target,
                 custom_tooltips=tooltip_s)
# Tooltips with the target y-labels (i.e. 1,2,3,...,9) for every cluster member
mapper.visualize(graph,
                 title="Handwritten digits Mapper",
                 path_html="digits_ylabel_tooltips.html",
                 custom_tooltips=digits.target)
