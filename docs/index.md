**Contributors:** {% for member in site.github.contributors %}<a href="{{member.html_url}}"><img src="{{member.avatar_url}}" width="32" height="32"></a>{% endfor %}

## Introduction
The Wave Analysis Network (WAnet) is a neural network for producing wave-structure interaction forces for ocean applciations. This repository contains all software required to recreate the networks.

## Table of Contents
* [The Basics]()
  * [NEMOH]()
  * [Neural Networks]()
* [Data]()
* [Training]()
  * [Training the Autoencoders]()
  * [Training the Analytical/Synthetic Networks]()
* [Application]()

## Acknowledgements
This work has been submitted to the [Design Computing and Cognition Conference](http://dccconferences.org/dcc18/).
*Portions of this repository (specifically, those in [openwec.py](https://github.com/HSDL/WAnet/tree/master/WAnet/openwec.py)) are based on [openWEC](https://github.com/tverbrug/openWEC)*
