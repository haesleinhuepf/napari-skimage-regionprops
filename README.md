# napari-skimage-regionprops (nsr)

[![License](https://img.shields.io/pypi/l/napari-skimage-regionprops.svg?color=green)](https://github.com/haesleinhuepf/napari-skimage-regionprops/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-skimage-regionprops.svg?color=green)](https://pypi.org/project/napari-skimage-regionprops)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-skimage-regionprops.svg?color=green)](https://python.org)
[![tests](https://github.com/haesleinhuepf/napari-skimage-regionprops/workflows/tests/badge.svg)](https://github.com/haesleinhuepf/napari-skimage-regionprops/actions)
[![codecov](https://codecov.io/gh/haesleinhuepf/napari-skimage-regionprops/branch/master/graph/badge.svg)](https://codecov.io/gh/haesleinhuepf/napari-skimage-regionprops)
[![Development Status](https://img.shields.io/pypi/status/napari-skimage-regionprops.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-skimage-regionprops)](https://napari-hub.org/plugins/napari-skimage-regionprops)


A [napari] plugin for measuring properties of labeled objects based on [scikit-image]

![](https://github.com/haesleinhuepf/napari-skimage-regionprops/raw/master/images/interactive.gif)

## Usage

From the menu `Tools > Measurement > Regionprops (nsr)` you can open a dialog where you can choose an intensity image, a corresponding label image and the features you want to measure:

![img.png](https://github.com/haesleinhuepf/napari-skimage-regionprops/raw/master/images/dialog.png)

If you want to interface with the labels and see which table row corresponds to which labeled object, use the label picker and
activate the `show selected` checkbox.

![](https://github.com/haesleinhuepf/napari-skimage-regionprops/raw/master/images/interactive.png)

If you closed a table and want to reopen it, you can use the menu `Tools > Measurements > Show table (nsr)` to reopen it. 
You just need to select the labels layer the properties are associated with.

For visualizing measurements with different grey values, as parametric images, you can use the menu `Tools > Visualization > Measurements on labels (nsr)`. 
After performing measurements, use this dialog to select the layer where measurements were performed on and layer properties were stored.
Also enter which column should be visualized.

![img.png](https://github.com/haesleinhuepf/napari-skimage-regionprops/raw/master/images/parametric_images.png)


## Usage, programmatically

You can also control the tables programmatically. See this [example notebook](https://github.com/haesleinhuepf/napari-skimage-regionprops/blob/master/demo/tables.ipynb) for details.

## Features
The user can select categories of features for feature extraction in the user interface. These categories contain measurements from the scikit-image [regionprops list of measurements](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops) library:
* size:
  * area
  * bbox_area
  * convex_area
  * equivalent_diameter
* intensity:
  * max_intensity 
  * mean_intensity
  * min_intensity
  * standard_deviation_intensity (`extra_properties` implementation using numpy)
* perimeter:
  * perimeter
  * perimeter_crofton
* shape
  * major_axis_length
  * minor_axis_length
  * orientation
  * solidity
  * eccentricity
  * extent
  * feret_diameter_max
  * local_centroid
* position:
  * centroid
  * bbox
  * weighted_centroid
* moments:
  * moments
  * moments_central
  * moments_hu
  * moments_normalized

This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

## Installation

You can install `napari-skimage-regionprops` via [pip]:

    pip install napari-skimage-regionprops

Or if you plan to develop it:

    git clone https://github.com/haesleinhuepf/napari-skimage-regionprops
    cd napari-skimage-regionprops
    pip install -e .

If there is an error message suggesting that git is not installed, run `conda install git`.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-skimage-regionprops" is free and open source software

## Issues

If you encounter any problems, please create a thread on [image.sc] along with a detailed description and tag [@haesleinhuepf].

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[image.sc]: https://image.sc
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
[scikit-image]: https://scikit-image.org/
[@haesleinhuepf]: https://twitter.com/haesleinhuepf
