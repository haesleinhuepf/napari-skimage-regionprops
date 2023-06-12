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

## Usage: measure region properties

From the menu `Tools > Measurement tables > Regionprops (nsr)` you can open a dialog where you can choose an intensity image, a corresponding label image and the features you want to measure:

![img.png](https://github.com/haesleinhuepf/napari-skimage-regionprops/raw/master/images/dialog.png)

Note that perimeter measurements are only supported for 2D images.

If you want to interface with the labels and see which table row corresponds to which labeled object, use the label picker and
activate the `show selected` checkbox.

![](https://github.com/haesleinhuepf/napari-skimage-regionprops/raw/master/images/interactive.png)

If you closed a table and want to reopen it, you can use the menu `Tools > Measurements > Show table (nsr)` to reopen it. 
You just need to select the labels layer on the basis of which the table was generated for the first time.

For visualizing measurements with different grey values as parametric images, you can double-click table headers.

![img.png](https://github.com/haesleinhuepf/napari-skimage-regionprops/raw/master/images/label_value_visualization.gif)

## Usage: measure point intensities

Analogously, also the intensity and coordinates of point layers can be measured using the menu `Tools > Measurement > Measure intensity at point coordinates (nsr)`. 
Also these measurements can be visualized by double-clicking table headers:

![img.png](measure_point_intensity.png)

![img_1.png](measure_point_coordinate.png)

## Working with time-lapse and tracking data

You can also derive measurements from a timelapse dataset by selecting `Tools > Measurement tables > Regionprops of all frames (nsr)`. If you do so, a "frame" column will be introduced. This column indicates which slice in
time the given row refers to.
If you want to import your own csv files for time-lapse data make sure to include this "frame" column as well.
If you have tracking data where each column specifies measurements for a track instead of a label at a specific time point,
this column must not be added.

In case you have 2D time-lapse data you need to convert it into a suitable shape using the function: `Tools > Utilities > Convert 3D stack to 2D time-lapse (time-slicer)`,
which can be found in the [napari time slicer](https://www.napari-hub.org/plugins/napari-time-slicer).

## Usage: multichannel or multi-label data

If you want to relate objects from one channels to objects from another channel, you can use `Tools > Measurement tables > Object Features/Properties (scikit-image, nsr)`. 
This plugin module allows you to answer questions like:
  - How many objects I have inside other objects?
  - What is the average intensity of the objects inside other objects?
For that, you need at least two labeled images in napari. You can relate objects along with their features. 
If intensity features are also wanted, then you also need to provide two intensity images. 
Below, there is a small example on how to use it. 
Also, take a look at [this example notebook](https://github.com/haesleinhuepf/napari-skimage-regionprops/blob/master/demo/measure_relationship_to_other_channels_plugin.ipynb).
 
 ![](https://github.com/haesleinhuepf/napari-skimage-regionprops/raw/master/images/things_inside_things_demo.gif)

## Usage, programmatically

You can also control the tables programmatically. See this 
[example notebook](https://github.com/haesleinhuepf/napari-skimage-regionprops/blob/master/demo/tables.ipynb) for details on regionprops and
[this example notebook](https://github.com/haesleinhuepf/napari-skimage-regionprops/blob/master/demo/measure_points.ipynb) for details on measuring intensity at point coordinates. For creating parametric map images, see [this notebook](https://github.com/haesleinhuepf/napari-skimage-regionprops/blob/master/demo/map_measurements.ipynb).


## Features
The user can select categories of features for feature extraction in the user interface. These categories contain measurements from the scikit-image [regionprops list of measurements](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops) library:
* size:
  * area (given as number of pixels in 2D, voxels in 3D)
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
  * roundness as defined for 2D labels [by ImageJ](https://imagej.nih.gov/ij/docs/menus/analyze.html#set)
  * circularity as defined for 2D labels  [by ImageJ](https://imagej.nih.gov/ij/docs/menus/analyze.html#set)
  * aspect_ratio as defined for 2D labels [by ImageJ](https://imagej.nih.gov/ij/docs/menus/analyze.html#set)
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

## See also

There are other napari plugins with similar functionality for extracting features:
* [morphometrics](https://www.napari-hub.org/plugins/morphometrics)
* [PartSeg](https://www.napari-hub.org/plugins/PartSeg)
* [napari-simpleitk-image-processing](https://www.napari-hub.org/plugins/napari-simpleitk-image-processing)
* [napari-cupy-image-processing](https://www.napari-hub.org/plugins/napari-cupy-image-processing)
* [napari-pyclesperanto-assistant](https://www.napari-hub.org/plugins/napari-pyclesperanto-assistant)

Furthermore, there are plugins for postprocessing extracted measurements
* [napari-feature-classifier](https://www.napari-hub.org/plugins/napari-feature-classifier)
* [napari-clusters-plotter](https://www.napari-hub.org/plugins/napari-clusters-plotter)
* [napari-accelerated-pixel-and-object-classification](https://www.napari-hub.org/plugins/napari-accelerated-pixel-and-object-classification)

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
