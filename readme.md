# Tone Mapping in CUDA

## Introduction

The tone mapping is a process in which the global contrast is reduced in order to enhance the local contrasts. 

One purpose of this technique, the objective of this project, is translate image in HDR (high dynamical range) to SDR (standard dynamic range) reducing the losses in quality that a linear compression in the luminance produces.

This algorithm extracts the luminance of the image and calculates its minimum, its maximum and the range of the luminance. With this data it has to calculate an histogram of the luminance of each pixel and then, it also calculates the normalize cumulative distribution function (CDF) of the histogram. Now we have the pixels distributed in "n" bins of different luminance. The last step is rebuild the RGB image with the new luminance. All these steps are been made in GPU using CUDA.


## Usage

Open the executable file in console with the following command: `./ToneMapping.exe input_file [output_filename]`, where `output_filename` is optional.


## Requeriments

This project needs CUDA and OpenCV. OpenCV 4.2.0 libraries and headers are included in the project. See [OpenCV project](https://opencv.org/) for more information.

## Questions

Write to the mail: r.lozanoc93@gmail.com.