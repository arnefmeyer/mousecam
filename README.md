[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![HitCount](http://hits.dwyl.io/arnefmeyer/mousecam.svg)](http://hits.dwyl.io/arnefmeyer/mousecam)

# mousecam

(click [here](https://arnefmeyer.github.io/mousecam) for a html version)

<p align="center">
<img src="docs/images/filmstrip_mouse.jpg" width="90%">
</p>

3D models and data extraction/analysis code for the head-mounted camera system described in:

>AF Meyer, J Poort, J O'Keefe, M Sahani, and JF Linden: _A head-mounted camera system integrates detailed behavioral monitoring with multichannel electrophysiology in freely moving mice_, Neuron, Volume 100, p46-60, 2018. [link (open access)](https://doi.org/10.1016/j.neuron.2018.09.020)

This repository contains the following:

* **mousecam**: a [Python package](https://github.com/arnefmeyer/mousecam/tree/master/mousecam) with functions for extracting and analyzing data recorded using the camera system, including
    - pupil fitting (including GUI)
    - body tracking (including GUI)
    - image registration (via fiji/imagej)
    - extraction of head orientation
* **parts**: [3D models](https://github.com/arnefmeyer/mousecam/tree/master/parts) (openscad/stl) and building instructions for the head-mounted camera system


## News

- 29/07/2021: Another paper using the mousecam published in [Nature Communications](https://www.nature.com/articles/s41467-021-24311-5)
- 08/06/2020: Paper on head and eye movements in freely moving mice published in [Current Biology](https://www.cell.com/current-biology/fulltext/S0960-9822(20)30556-X)
- 20/02/2020: New preprint using the mousecam on [bioRxiv](https://biorxiv.org/cgi/content/short/2020.02.20.957712v1)
- 21/11/2018: Add design files for [new camera module](https://github.com/arnefmeyer/mousecam/tree/master/parts)
- 04/11/18: This page is now also organized in a [website format](https://arnefmeyer.github.io/mousecam).
- 07/11/18: Added websites with step-by-step building instructions
- 28/11/18: The project is now on [open-ephys.org](http://www.open-ephys.org/mousecam/)
- 12/01/19: Added intructions on [how to remove IR filter from camera module](docs/building_instructions.md)
- 14/02/19: Mousecam featured on [openbehavior.com](https://edspace.american.edu/openbehavior/2019/02/06/head-mounted-camera-system/)


## Step-by-step building instructions

<p align="center">
<img src="docs/images/mousecam_3dparts.png" width="85%">
</p>

[1. Parts list](docs/parts_list.md)

[2. Building instructions](docs/building_instructions.md)

[3. Implantation](docs/implantation.md)

[4. Calibration](docs/calibration.md)


## Software

### Head-mounted camera code

[Custom camera software](https://github.com/arnefmeyer/RPiCameraPlugin) for the Raspberry Pi (RPi), which also integrates with the open-ephys plugin-GUI. As this software uses [zeromq](http://zeromq.org/) (with [bindings](http://zeromq.org/bindings:_start) for many programming languages) for communication between the recording computer and the RPi, it can easily be extended/adapted to other systems.


### Measurement of head orientation/movement data

Code (including open-ephys plugin) for reading data from intertia measurement unit (IMU) sensors and synchronizing movement data with neural recordings is available [here](https://github.com/arnefmeyer/IMUReaderPlugin).


### Behavioral segmentation

A Python package for behavioral scoring (including GUI for manual annotation) can be found [here](https://github.com/arnefmeyer/BehavioralScoring).


## Contribute

If you are looking for a different variant of the design or want to contribute modified and/or new designs, you can find all files in the [3D model directory](https://github.com/arnefmeyer/mousecam/tree/master/parts). Please use the *issue tracker* to report problems with building the camera system and create *pull requests* for improved/new designs.

- Issue Tracker: https://github.com/arnefmeyer/mousecam/issues
- Source Code: https://github.com/arnefmeyer/mousecam
- Project Website: https://arnefmeyer.github.io/mousecam


## References

```
@Article{Meyer2018,
  author    = {Meyer, Arne F. and Poort, Jasper and O’Keefe, John and Sahani, Maneesh and Linden, Jennifer F.},
  title     = {A Head-Mounted Camera System Integrates Detailed Behavioral Monitoring with Multichannel Electrophysiology in Freely Moving Mice},
  journal   = {Neuron},
  year      = {2018},
  volume    = {100},
  number    = {1},
  month     = oct,
  pages     = {46--60},
  issn      = {0896-6273},
  doi       = {10.1016/j.neuron.2018.09.020},
  url       = {https://doi.org/10.1016/j.neuron.2018.09.020},
  publisher = {Elsevier},
}

@Article{Meyer2020,
  author    = {Meyer, Arne F. and O'Keefe, John and Poort, Jasper},
  title     = {Two Distinct Types of Eye-Head Coupling in Freely Moving Mice},
  doi       = {10.1016/j.cub.2020.04.042},
  issn      = {0960-9822},
  number    = {11},
  pages     = {2116--2130},
  url       = {https://doi.org/10.1016/j.cub.2020.04.042},
  volume    = {30},
  comment   = {doi: 10.1016/j.cub.2020.04.042},
  journal   = {Current Biology},
  month     = jun,
  publisher = {Elsevier},
  year      = {2020},
}
```
