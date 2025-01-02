# mbcs2025

Code repository from "Learned spectral and spatial transforms for multispectral remote sensing data compression", by S. Mijares, J. Bartrina-Rapesta, M. Hernández-Cabronero, and J. Serra-Sagristà, submitted to the IEEE Geoscience and Remote Sensing Letters in 2025. This repository contains the scripts to train and run the proposed models. The following ReadMe details how to use the code, where to access test data, and the paper's pre trained models.

## Test data

The publically available sources of test data used for this paper, Landsat 8 OLI, AVIRIS, and Sentinel 2A, are available in the [GICI website](https://gici.uab.cat/GiciWebPage/datasets.php).

Data is stored in .raw format as 16-bit unsigned samples, little endian byte order, and BSQ sample order. Test data files are stored with a standardised filename that the architecture script is automated to read:

```<image name>.<bands>_<width>_<height>_<data type>_<endianness>_<is it RGB?>.raw```

The data type is 1 for 8-bit data (unsigned), 2 for 16-bit unsigned integers, 3 for 16-bit signed integers, 4 for 32-bit integers, and 6 32-bit floating point numbers. If an image does not use this naming format, the user must specify these values in their command.

## Models

Pre-trained models to generate the results in this paper are available in the [GICI website](https://gici.uab.cat/GiciWebPage/downloads.php).

## Training a model

To train a model, call the `train` command in the architecture script using the corresponding options. For example:

```python3 architecture.py --model_path ./models/model_name train "/path/to/folder/*.raw" --lambda 0.00001 0.01 --epochs 1000 --steps_per_epoch 1000 --patchsize 256 --batchsize 6 --height 512 --width 512 --bands 8```

As described in the main paper, the loss function is:

$L = R(y)+ \lambda_1 MSE(x_{1D},\hat{x}_{1D}) + \frac{1}{(\det A)^2} + (\det A)^2 + \lambda_2 MSE( A^T A, I_n) + \lambda_3 MSE(x_{i,j}, A^{-1}P_k A x_{i,j})$

The following table lists $\lambda$ values used for training our models.

| Parameter  | Landsat 8          | Sentinel 2          | AVIRIS bands 42-48 | Worldview 3    |
| ---------- | ------------------ | ------------------- | ------------------ | -------------- |
| $\lambda_1$ | $[0.00001, 0.001]$ |  $[0.00001, 0.001]$ | $[0.0001, 0.1]$    | $[0.001, 0.1]$ |
| $\lambda_2$ | 100                | 100                 | 100                | 100            |
| $\lambda_3$ | $10^{-13}$         | $10^{-10}$          | $10^{-13}$         | $10^{-9}$      |

## Running a trained model

To use a trained model for compression and decompression, use the `architecture.py` script. For compression using a quality parameter, the command is as follows:

```python3 architecture.py --model_path ./models/some_model_folder compress /path/to/folder/image.8_512_512_2_1_0.raw --quality 0.001```

You can also compress multiple images in one run using wildcard inputs. This is faster than running the script multiple times, as the compilation, imprting of libraries (tensorflow and tensorflow-compression in particular), and loading of the model will be done only once, substatially speeding up runtime. Make sure you write the input filename in quotation marks in such case:

```python3 architecture.py --model_path ./models/some_model_folder compress "/path/to/folder/*.raw" --quality 0.001```

This architecture incorporates the compression at a user-defined bit rate feature from

> S. M. i. Verdú, M. Chabert, T. Oberlin and J. Serra-Sagristà, "Reduced-Complexity Multirate Remote Sensing Data Compression With Neural Networks," in IEEE Geoscience and Remote Sensing Letters, vol. 20, pp. 1-5, 2023, Art no. 6011705, doi: 10.1109/LGRS.2023.3325477.

This can be done as follows:

```python3 architecture.py --model_path ./models/some_model_folder compress /path/to/folder/image.8_512_512_2_1_0.raw --bitrate 0.001```

To decompress any compressed image (or set of images), simply use the `decompress` command:

```python3 architecture.py --model_path ./models/some_model_folder decompress /path/to/folder/image.8_512_512_2_1_0.raw.tfci```
