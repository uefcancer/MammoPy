.. raw:: html

   <p align="center">

.. raw:: html

   </p>

MammoPy
=======

A Comprehensive Deep Learning Library for Mammogram Assessment

|PyPI version| |GitHub| # Useful Links
`[Documentation] <https://uefcancer.github.io/MammoPy/>`__ \|
`[Paper] <https://www.nature.com/articles/s41598-021-93169-w.pdf>`__ \|
`[Notebook
examples] <https://github.com/uefcancer/MammoPy/tree/main/notebooks>`__
\| `[Web applications] <https://wiki-breast.onrender.com/>`__ #
Introduction **Welcome to ``MammoPy`` Repository!** ``MammoPy`` is a
python-based library designed to facilitate the creation of mammogram
image analysis pipelines . The library includes plug-and-play modules to
perform:

-  Standard mammogram image pre-processing (e.g., *normalization*,
   *bounding box cropping*, and *DICOM to jpeg conversion*)

-  Mammogram assessment pipelines (e.g., *breast area segmentation*,
   *dense tissue segmentation*, and *percentage density estimation*)

-  Modeling deep learning architectures for various downstream tasks
   (e.g., *micro-calcification* and *mass detection*)

-  Feature attribution-based interpretability techniques (e.g.,
   *GradCAM*, *GradCAM++*, and *LRP*)

-  Visualization

All the functionalities are grouped under a user-friendly API.

If you encounter any issue or have questions regarding the library, feel
free to `open a GitHub
issue <https://github.com/uefcancer/mammopy/issues>`__. We’ll do our
best to address it.

.. |PyPI version| image:: https://badge.fury.io/py/mammopy.svg
   :target: https://badge.fury.io/py/mammopy
.. |GitHub| image:: https://img.shields.io/github/license/mammopy/mammopy
