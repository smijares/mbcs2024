#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D+2D multi-rate neural network codec with linear reversible, hierarchical and orthogonal spectral transform.
V3.5
Sebastià Mijares i Verdú - GICI, UAB
sebastia.mijares@uab.cat

Neural network archiecture from "Learned spectral and spatial transforms for multispectral remote sensing data compression", by S. Mijares, J. Bartrina-Rapesta, M. Hernández-Cabronero, and J. Serra-Sagristà.

Hard-coded parameters
---------------------

data_type: data type of .raw data the network will be used on.
bitrate_precision: bitrate precision for fixed-rate compression.
bit_length: bit depth of the data the network will be used (ex: 12-bit data stored as 16-bit samples).

Requirements
------------

argparse
glob
sys
absl
tensorflow
tensorflow-compression
numpy
os

Desenvolupament
---------------

"""

import argparse
import glob
import sys
from absl import app
from absl.flags import argparse_flags
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow_compression as tfc
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

"""
WARNING: THIS CODE WILL ONLY RUN FOR ONE DATATYPE (uint8, uint16, etc.).

THIS IS HARD-CODED IN THE PARAMETERS BELOW.
"""

#Parameters

data_type = tf.uint16
bitrate_precision = 0.05
bit_length = 16

#Running code

def read_raw(filename, height, width, bands, endianess, DTYPE=data_type):
  """
  Reads a raw image file and returns a tensor of given height, width, and number of components, taking endianess and bytes-per-entry into account.

  This function is independent from the patchsize chosen for training.
  """
  string = tf.io.read_file(filename)
  vector = tf.io.decode_raw(string, DTYPE, little_endian=(endianess==1))
  return tf.transpose(tf.reshape(vector, [bands, height, width]), (1,2,0))

def write_raw(filename, image):
  """
  Saves an image to a raw file.
  """
  arr = np.transpose(np.array(image),(2,0,1))
  arr.tofile(filename, format='.raw')

def get_geometry(file):
    if file[-4:]=='.raw':
        G = file.split('.')[1].split('_')
        bands = G[0]
        width = G[1]
        height = G[2]
        datatype = G[3]
        endianess = G[4]
        #Note these are all strings, not int numbers!
        return (bands, width, height, endianess, datatype)
    print('No RAW files were found. Assumes 8-bit PNG files instead.')
    return ('0', '0', '0', '0', '1')

class ModulatingTransform(tf.keras.Sequential):
  """
  The modulating or demodulating transform.
  
  The lambda input value is normalised by the maximum lambda value provided in training.
  """

  def __init__(self, hidden_nodes, num_filters, maxval):
    super().__init__()
    self.add(tf.keras.layers.Lambda(lambda x: x / maxval))
    self.add(tf.keras.layers.Dense(hidden_nodes, activation=tf.nn.relu, kernel_initializer='ones'))
    self.add(tf.keras.layers.Dense(num_filters, activation=tf.nn.relu, kernel_initializer='ones'))

class AnalysisTransform(keras.Sequential):
  """The analysis transform."""

  def __init__(self, num_filters_hidden, num_filters_latent):
    super().__init__(name="analysis")
    self.add(keras.layers.Lambda(lambda x: x / ((2**bit_length)-1)))
    self.add(tfc.SignalConv2D(
        num_filters_hidden, (5, 5), name="layer_0", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_0")))
    self.add(tfc.SignalConv2D(
        num_filters_hidden, (5, 5), name="layer_1", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_1")))
    self.add(tfc.SignalConv2D(
        num_filters_hidden, (5, 5), name="layer_2", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_1")))
    self.add(tfc.SignalConv2D(
        num_filters_latent, (5, 5), name="layer_3", corr=True, strides_down=2,
        padding="same_zeros", use_bias=False,
        activation=None))


class SynthesisTransform(keras.Sequential):
  """The synthesis transform."""

  def __init__(self, num_filters):
    super().__init__(name="synthesis")
    self.add(keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_0", corr=False, strides_up=1,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_0", inverse=True)))
    self.add(keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=False, strides_up=1,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_1", inverse=True)))
    self.add(keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_2", corr=False, strides_up=1,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_2", inverse=True)))
    self.add(keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
    self.add(tfc.SignalConv2D(
        1, (5, 5), name="layer_3", corr=False, strides_up=1,
        padding="same_zeros", use_bias=True,
        activation=None))
    self.add(keras.layers.Lambda(lambda x: x * ((2**bit_length)-1)))


class HyperAnalysisTransform(keras.Sequential):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, num_filters_hidden_hyperprior, num_filters_latent_hyperprior):
    super().__init__(name="hyper_analysis")
    self.add(tfc.SignalConv2D(
        num_filters_hidden_hyperprior, (3, 3), name="layer_0", corr=True, strides_down=1,
        padding="same_zeros", use_bias=True,
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters_hidden_hyperprior, (5, 5), name="layer_1", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters_latent_hyperprior, (5, 5), name="layer_2", corr=True, strides_down=2,
        padding="same_zeros", use_bias=False,
        activation=None))

class HyperSynthesisTransform(keras.Sequential):
  """The synthesis transform for the entropy model parameters."""

  def __init__(self, num_filters_hidden_hyperprior, num_filters_latent):
    super().__init__(name="hyper_synthesis")
    self.add(keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
    self.add(tfc.SignalConv2D(
        num_filters_hidden_hyperprior, (5, 5), name="layer_0", corr=False, strides_up=1,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=tf.nn.relu))
    self.add(keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
    self.add(tfc.SignalConv2D(
        num_filters_hidden_hyperprior, (5, 5), name="layer_1", corr=False, strides_up=1,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters_latent, (3, 3), name="layer_2", corr=False, strides_up=1,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=None))
    
class SpectralAnalysisTransform(keras.Sequential):
  """The analysis transform."""

  def __init__(self, num_filters_1D, init):
    super().__init__(name="analysis")
    self.add(keras.layers.Dense(num_filters_1D, activation=None, use_bias=False, kernel_initializer=init))


class CSMR2023061401_v4_0(keras.Model):
  """Main model class."""

  def __init__(self, lmbda, num_filters, num_scales, scale_min, scale_max, bands, progressive, initialisation):
    super().__init__()
    self.max_lambda = lmbda[1] #MSE lambda (minimum)
    self.min_lambda = lmbda[0] #MSE lambda (maximum)
    self.lmbda3 = lmbda[2] #regulariser
    self.lmbda4 = lmbda[3] #progressive variance
    self.num_scales = num_scales
    self.bands = bands
    self.num_filters_latent = num_filters[1]
    if initialisation == None:
        init = 'identity'
    else:
        weights = np.reshape(np.fromfile(initialisation, dtype=np.float32), (bands, bands))
        init = tf.constant_initializer(weights)
    offset = tf.math.log(scale_min)
    factor = (tf.math.log(scale_max) - tf.math.log(scale_min)) / (
        num_scales - 1.)
    if len(num_filters) == 1:
        num_filters.append(num_filters[0])
    self.scale_fn = lambda i: tf.math.exp(offset + factor * i)
    self.analysis_transform = AnalysisTransform(num_filters[0], num_filters[1])
    self.synthesis_transform = SynthesisTransform(num_filters[0])
    self.spectral_analysis_transform = SpectralAnalysisTransform(bands, init)
    self.hyper_analysis_transform = HyperAnalysisTransform(num_filters[2], num_filters[3])
    self.hyper_synthesis_transform = HyperSynthesisTransform(num_filters[2], num_filters[1]*bands)
    self.modulating_transform = ModulatingTransform(192, num_filters[1]*bands, lmbda[1])
    self.hyperprior = tfc.NoisyDeepFactorized(batch_shape=(num_filters[3],))
    self.minband = progressive[0]
    self.maxband = progressive[1]
    self.stepsize = progressive[2]
    if self.maxband >= bands:
    	self.maxband = bands
    if self.minband >= self.maxband:
    	self.minband = self.maxband
    if self.stepsize == 0:
    	self.stepsize = 1
    self.build((None, None, None, bands))

  def call(self, x, training):
    """Computes rate and distortion losses."""
    entropy_model = tfc.LocationScaleIndexedEntropyModel(
        tfc.NoisyNormal, self.num_scales, self.scale_fn, coding_rank=3,
        compression=False)
    side_entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.hyperprior, coding_rank=3, compression=False)

    x_spec = self.spectral_analysis_transform(x)
    x_1D = tf.transpose(x_spec,(0,3,1,2))
    x_1D = tf.reshape(x_1D,(tf.shape(x)[0]*self.bands, 1, tf.shape(x)[1], tf.shape(x)[2]))
    x_1D = tf.transpose(x_1D,(0,2,3,1))
    
    lmda = tf.random.uniform((1,), minval=self.min_lambda, maxval=self.max_lambda)
    
    y = self.analysis_transform(x_1D)
    mod = self.modulating_transform(tf.expand_dims(tf.expand_dims(tf.expand_dims(lmda, 0), 0), 0))
    
    y_3D = tf.transpose(y,(0,3,1,2))
    y_3D = tf.reshape(y_3D,(tf.shape(x)[0], self.num_filters_latent*self.bands, tf.shape(x)[1]//16, tf.shape(x)[2]//16))
    y_3D = tf.transpose(y_3D,(0,2,3,1))
    
    y_3D = mod*y_3D
    
    z = self.hyper_analysis_transform(abs(y_3D))
    z_hat, side_bits = side_entropy_model(z, training=training)
    indexes = self.hyper_synthesis_transform(z_hat)
    y_hat_3D, bits = entropy_model(y_3D, indexes, training=training)
    
    demod = 1/mod
    y_hat_3D = demod*y_hat_3D
    
    y_hat_3D = tf.transpose(y_hat_3D,(0,3,1,2))
    y_hat_3D = tf.reshape(y_hat_3D,(tf.shape(x)[0]*self.bands, self.num_filters_latent, tf.shape(x)[1]//16, tf.shape(x)[2]//16))
    y_hat = tf.transpose(y_hat_3D,(0,2,3,1))
    
    x_hat_1D = self.synthesis_transform(y_hat)
    
    I = tf.expand_dims(tf.expand_dims(tf.eye(self.bands), axis=1), axis=1)
    A = tf.squeeze(self.spectral_analysis_transform(I))
    det = tf.linalg.det(A)
    T = tf.linalg.matrix_transpose(A)
    reg = tf.reduce_sum((tf.linalg.matmul(A,T)-tf.squeeze(I))**2)
    
    # In this version, the progressive hierarchisation loss is calculated only using the spectral transfrom, not the full transform.
    k = self.minband + self.stepsize*tf.squeeze(tf.random.categorical(tf.math.log([((self.maxband-(self.minband))//self.stepsize)*[1/((self.maxband-(self.minband))//self.stepsize),]]), 1))
    x_speck = tf.concat((x_spec[:, :, :, :k],0*x_spec[:, :, :, k:]), axis=-1)
    part_loss = tf.reduce_sum((tf.linalg.matvec(T,x_speck)-x)**2)

    # Total number of bits divided by total number of pixels.
    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), bits.dtype)
    bpp = (tf.reduce_sum(bits) + tf.reduce_sum(side_bits)) / num_pixels
    # Mean squared error across pixels.
    mse_2D = tf.reduce_mean(tf.math.squared_difference(x_1D, x_hat_1D))
    # The rate-distortion Lagrangian.
    loss = bpp + lmda * mse_2D + 1/(det**2) + det**2 + self.lmbda3*reg + self.lmbda4*part_loss
    return loss, bpp, mse_2D, det, reg, part_loss

  def train_step(self, x):
    with tf.GradientTape() as tape:
      loss, bpp, mse_2D, det, reg, part_loss = self(x, training=True)
    variables = self.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse_2D.update_state(mse_2D)
    self.det.update_state(det)
    self.reg.update_state(reg)
    self.part_loss.update_state(part_loss)
    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse_2D, self.det, self.reg, self.part_loss]}

  def test_step(self, x):
    loss, bpp, mse_2D, det, reg, part_loss = self(x, training=False)
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse_2D.update_state(mse_2D)
    self.det.update_state(det)
    self.reg.update_state(reg)
    self.part_loss.update_state(part_loss)
    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse_2D, self.det, self.reg, self.part_loss]}

  def predict_step(self, x):
    raise NotImplementedError("Prediction API is not supported.")

  def compile(self, **kwargs):
    super().compile(
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        **kwargs,
    )
    self.loss = keras.metrics.Mean(name="loss")
    self.bpp = keras.metrics.Mean(name="bpp")
    self.mse_2D = keras.metrics.Mean(name="mse_2D")
    self.det = keras.metrics.Mean(name="det")
    self.reg = keras.metrics.Mean(name="regulariser")
    self.part_loss = keras.metrics.Mean(name="hierarchisation")

  def fit(self, *args, **kwargs):
    retval = super().fit(*args, **kwargs)
    # After training, fix range coding tables.
    self.entropy_model = tfc.LocationScaleIndexedEntropyModel(
        tfc.NoisyNormal, self.num_scales, self.scale_fn, coding_rank=3,
        compression=True)
    self.side_entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.hyperprior, coding_rank=3, compression=True)
    return retval

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, None), dtype=data_type)
  ])
  def main_transform(self, x):
    """Applies analysis transform and returns y vector."""
    # Add batch dimension and cast to float.
    x = tf.cast(x, dtype=tf.float32)
    x = tf.expand_dims(x, 0)
    x_shape = tf.shape(x)[1:-1]
    
    x_1D = self.spectral_analysis_transform(x)
    x_1D = tf.transpose(x_1D,(0,3,1,2))
    x_1D = tf.reshape(x_1D,(self.bands, 1, x_shape[0], x_shape[1]))
    x_1D = tf.transpose(x_1D,(0,2,3,1))
    y = self.analysis_transform(x_1D)
    
    y_shape = tf.shape(y)[1:-1]
    return y, x_shape, y_shape

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32)
  ])
  def modulation(self, y, x_shape, y_shape, lmda):
    """Applies analysis transform and returns y vector."""
    # Add batch dimension and cast to float.
    lmda = tf.maximum(tf.minimum(lmda, self.max_lambda), self.min_lambda)
    lmda_send = tf.round(255.*(lmda-self.min_lambda)/(self.max_lambda-self.min_lambda))
    lmda_quant = (lmda_send/255.)*(self.max_lambda-self.min_lambda)+self.min_lambda
    mod = self.modulating_transform(lmda_quant)
    
    y_3D = tf.transpose(y,(0,3,1,2))
    y_3D = tf.reshape(y_3D,(1, self.num_filters_latent*self.bands, x_shape[0]//16, x_shape[1]//16))
    y_3D = tf.transpose(y_3D,(0,2,3,1))
    
    y_3D = mod*y_3D
    
    z = self.hyper_analysis_transform(abs(y_3D))
    # Preserve spatial shapes of both image and latents.
    z_shape = tf.shape(z)[1:-1]
    z_hat, _ = self.side_entropy_model(z, training=False)
    indexes = self.hyper_synthesis_transform(z_hat)
    indexes = indexes[:, :y_shape[0], :y_shape[1], :]
    side_string = self.side_entropy_model.compress(z)
    string = self.entropy_model.compress(y_3D, indexes)
    bps = tf.cast((tf.strings.length(string)+tf.strings.length(side_string))*8/(tf.reduce_prod(x_shape)*self.bands), dtype=tf.float32)
    return string, side_string, z_shape, tf.cast(lmda_send, tf.uint8), bps

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None,), dtype=tf.string),
      tf.TensorSpec(shape=(None,), dtype=tf.string),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(None, None, None, None), dtype=tf.uint8)
  ])
  def decompress(self, string, side_string, x_shape, y_shape, z_shape, L):
    """Decompresses an image."""
    I = tf.expand_dims(tf.expand_dims(tf.eye(self.bands), axis=1), axis=1)
    A = tf.squeeze(self.spectral_analysis_transform(I))
    B = tf.linalg.inv(A)
    
    V = (tf.cast(L, tf.float32)/255.)*(self.max_lambda-self.min_lambda)+self.min_lambda
    
    demod = 1/self.modulating_transform(V)
    z_hat = self.side_entropy_model.decompress(side_string, z_shape)
    indexes = self.hyper_synthesis_transform(z_hat)
    indexes = indexes[:, :y_shape[0], :y_shape[1], :]
    y_hat_3D = self.entropy_model.decompress(string, indexes)
    
    y_hat_3D = demod*y_hat_3D
    
    y_hat_3D = tf.transpose(y_hat_3D,(0,3,1,2))
    y_hat_3D = tf.reshape(y_hat_3D,(self.bands, self.num_filters_latent, x_shape[0]//16, x_shape[1]//16))
    y_hat = tf.transpose(y_hat_3D,(0,2,3,1))
    
    x_hat_1D = self.synthesis_transform(y_hat)
    x_hat_1D = tf.transpose(x_hat_1D,(0,3,1,2))
    x_hat_1D = tf.reshape(x_hat_1D,(1, self.bands, x_shape[0], x_shape[1]))
    x_hat_1D = tf.transpose(x_hat_1D,(0,2,3,1))
    x_hat = tf.linalg.matvec(tf.linalg.matrix_transpose(B), x_hat_1D)
    # Remove batch dimension, and crop away any extraneous padding.
    x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]
    # Then cast back to 8-bit integer.
    return tf.saturate_cast(tf.saturate_cast(tf.round(x_hat), tf.int32), data_type)
  
  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, None), dtype=data_type),
  ])
  def compress1D(self, x):
    """Compresses an image."""
    # Add batch dimension and cast to float.
    x = tf.cast(x, dtype=tf.float32)
    x = tf.expand_dims(x, 0)
    
    x_1D = self.spectral_analysis_transform(x)
    
    return tf.squeeze(x_1D)

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
  ])
  def decompress1D(self, y):
    """Compresses an image."""
    # Add batch dimension and cast to float.
    y = tf.expand_dims(y, 0)
    y_shape = tf.shape(y)[1:-1]
    
    I = tf.expand_dims(tf.expand_dims(tf.eye(self.bands), axis=1), axis=1)
    A = tf.squeeze(self.spectral_analysis_transform(I))
    B = tf.linalg.inv(A)
    
    x_hat = tf.linalg.matvec(tf.linalg.matrix_transpose(B), y)
    # Remove batch dimension, and crop away any extraneous padding.
    x_hat = x_hat[0, :y_shape[0], :y_shape[1], :]
    
    return tf.saturate_cast(tf.saturate_cast(tf.round(x_hat), tf.int32), data_type)

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32)
  ])
  def plain_modulation(self, lmda):
    """Applies analysis transform and returns y vector."""
    # Add batch dimension and cast to float.
    mod = self.modulating_transform(lmda)
    return mod

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
      tf.TensorSpec(shape=(2,), dtype=tf.int32)
  ])
  def inverse_transform(self, y_hat_3D, x_shape):
    """Decompresses an image only using the main transform, no decoding."""
    I = tf.expand_dims(tf.expand_dims(tf.eye(self.bands), axis=1), axis=1)
    A = tf.squeeze(self.spectral_analysis_transform(I))
    B = tf.linalg.inv(A)
    
    y_hat_3D = tf.transpose(y_hat_3D,(0,3,1,2))
    y_hat_3D = tf.reshape(y_hat_3D,(self.bands, self.num_filters_latent, x_shape[0]//16, x_shape[1]//16))
    y_hat = tf.transpose(y_hat_3D,(0,2,3,1))
    
    x_hat_1D = self.synthesis_transform(y_hat)
    x_hat_1D = tf.transpose(x_hat_1D,(0,3,1,2))
    x_hat_1D = tf.reshape(x_hat_1D,(1, self.bands, x_shape[0], x_shape[1]))
    x_hat_1D = tf.transpose(x_hat_1D,(0,2,3,1))
    x_hat = tf.linalg.matvec(tf.linalg.matrix_transpose(B), x_hat_1D)
    # Remove batch dimension, and crop away any extraneous padding.
    # Then cast back to data type.
    return tf.saturate_cast(tf.saturate_cast(tf.round(x_hat), tf.int32), data_type)


def check_image_size(image, patchsize, bands):
  shape = tf.shape(image)
  return shape[0] >= patchsize and shape[1] >= patchsize and shape[-1] == bands


def crop_image(image, patchsize, bands):
  image = tf.image.random_crop(image, (patchsize, patchsize, bands))
  return tf.cast(image, tf.float32)


def get_dataset(name, split, args):
  """Creates input data pipeline from a TF Datasets dataset."""
  with tf.device("/cpu:0"):
    dataset = tfds.load(name, split=split, shuffle_files=True)
    if split == "train":
      dataset = dataset.repeat()
    dataset = dataset.filter(
        lambda x: check_image_size(x["image"], args.patchsize))
    dataset = dataset.map(
        lambda x: crop_image(x["image"], args.patchsize))
    dataset = dataset.batch(args.batchsize, drop_remainder=True)
  return dataset


def get_custom_dataset(split, args):
  """Creates input data pipeline from custom RAW images."""
  with tf.device("/cpu:0"):
    files = glob.glob(args.train_glob)
    if not files:
      raise RuntimeError(f"No training images found with glob "
                         f"'{args.train_glob}'.")
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.shuffle(len(files), reshuffle_each_iteration=True)
    if split == "train":
      dataset = dataset.repeat()
    dataset = dataset.map(
        lambda x: crop_image(read_raw(x, args.height, args.width, args.bands, args.endianess), args.patchsize, args.bands),
        num_parallel_calls=args.preprocess_threads)
    dataset = dataset.batch(args.batchsize, drop_remainder=True)
  return dataset


def train(args):
  """Instantiates and trains the model."""
  if args.check_numerics:
    tf.debugging.enable_check_numerics()

  model = CSMR2023061401_v4_0(
      args.lmbda, args.num_filters, args.num_scales, args.scale_min,
      args.scale_max, args.bands, args.progressive, args.initialisation)
  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
  )

  if args.train_glob:
    train_dataset = get_custom_dataset("train", args)
    validation_dataset = get_custom_dataset("validation", args)
  else:
    train_dataset = get_dataset("clic", "train", args)
    validation_dataset = get_dataset("clic", "validation", args)
  validation_dataset = validation_dataset.take(args.max_validation_steps)
  
  config = tf.compat.v1.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.98 # fraction of memory
  config.gpu_options.visible_device_list = "0"
  tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

  model.fit(
      train_dataset.prefetch(8),
      epochs=args.epochs,
      steps_per_epoch=args.steps_per_epoch,
      validation_data=validation_dataset.cache(),
      validation_freq=1,
      callbacks=[
          keras.callbacks.TerminateOnNaN(),
          keras.callbacks.TensorBoard(
              log_dir=args.train_path,
              histogram_freq=1, update_freq="epoch"),
          keras.callbacks.experimental.BackupAndRestore(args.train_path),
      ],
      verbose=int(args.verbose),
  )
  model.save(args.model_path)


def compress(args):
  """Compresses an image."""
  # Load model and use it to compress the image.
  model = keras.models.load_model(args.model_path)
  inputs = glob.glob(args.input_file)
  for input_file in inputs:
      if args.height == None or args.width == None or args.bands == None:
          bands, width, height, endianess, datatype = get_geometry(input_file.split('/')[-1])
          x = read_raw(input_file,int(height),int(width),int(bands),int(endianess))
      else:
          x = read_raw(input_file,args.height,args.width,args.bands,args.endianess)
      y, x_shape, y_shape = model.main_transform(x)
      
      if args.quality_array:
          # Note the array is expected to be stored with the same endianness as the main array.
          Q = tf.expand_dims(read_raw(args.quality_array, int(height)//16, int(width)//16, 1, int(endianess), tf.float32),0)
      else:
          Q = tf.ones((1, 1, 1, 1), dtype=tf.float32)*args.quality

      string_max, side_string_max, z_shape, L_max, bps_max = model.modulation(y, x_shape, y_shape, Q)
      if args.bitrate == None:
          string, side_string, L = string_max, side_string_max, L_max
      else:
          string_min, side_string_min, z_shape, L_min, bps_min = model.modulation(y, x_shape, y_shape, tf.constant([0], dtype=tf.float32))
          max_lmbda = args.quality
          min_lmbda = 0
          iteration = 1
          while iteration < 101: #limited to 100 iterations to avoid infinite loops.
            iteration += 1
            if args.bitrate > bps_max-bitrate_precision or iteration == 100: #limited to 100 iterations to avoid infinite loops.
                string, side_string, L = string_max, side_string_max, L_max
                break
            elif args.bitrate < bps_min+bitrate_precision:
                string, side_string, L = string_min, side_string_min, L_min
                break
            mid_lmbda = (max_lmbda+min_lmbda)/2
            string_mid, side_string_mid, z_shape, L_mid, bps_mid = model.modulation(y, x_shape, y_shape, tf.constant([mid_lmbda]))        
            if bps_mid > args.bitrate:
                string_max, side_string_max, L_max, max_lmbda, bps_max = string_mid, side_string_mid, L_mid, mid_lmbda, bps_mid
            else:
                min_lmbda = mid_lmbda
                string_min, side_string_min, L_min, min_lmbda, bps_min = string_mid, side_string_mid, L_mid, mid_lmbda, bps_mid
      L = tf.reshape(L, (-1,))
      tensors = string, side_string, x_shape, y_shape, z_shape, L
    
      # Write a binary file with the shape information and the compressed string.
      packed = tfc.PackedTensors()
      packed.pack(tensors)
      with open(input_file+'.tfci', "wb") as f:
        f.write(packed.string)


def decompress(args):
  """Decompresses an image."""
  # Load the model and determine the dtypes of tensors required to decompress.
  model = keras.models.load_model(args.model_path)
  dtypes = [t.dtype for t in model.decompress.input_signature]
  inputs = glob.glob(args.input_file)
  for input_file in inputs:
      # Read the shape information and compressed string from the binary file,
      # and decompress the image using the model.
      with open(input_file, "rb") as f:
        packed = tfc.PackedTensors(f.read())
      string, side_string, x_shape, y_shape, z_shape, L = packed.unpack(dtypes)
      try:
          L = tf.reshape(L, (1, y_shape[0], y_shape[1], 1))
      except:
          L = tf.reshape(L, (1, 1, 1, 1))
      tensors = string, side_string, x_shape, y_shape, z_shape, L
      x_hat = model.decompress(*tensors)
    
      # Write reconstructed image out as a RAW file.
      write_raw(input_file+'.raw', x_hat)
      
def compress1D(args):
  """Compresses an image using only the spectral transform."""
  # Load model and use it to compress the image.
  model = keras.models.load_model(args.model_path)
  inputs = glob.glob(args.input_file)
  for input_file in inputs:
      if args.height == None or args.width == None or args.bands == None:
          bands, width, height, endianess, datatype = get_geometry(input_file.split('/')[-1])
          geometry = '.'+input_file.split('/')[-1].split('.')[-2]+'.raw'
          path = input_file.replace(geometry,'')
          x = read_raw(input_file,int(height),int(width),int(bands),int(endianess))
      else:
          x = read_raw(input_file,args.height,args.width,args.bands,args.endianess)
          path = input_file
      x1D = model.compress1D(x)

      write_raw(path+'_spectral-transform.'+bands+'_'+width+'_'+height+'_6_1_0.raw', x1D)

def decompress1D(args):
  """Deompresses an image using only the spectral transform (raw image input)."""
  # Load model and use it to compress the image.
  model = keras.models.load_model(args.model_path)
  inputs = glob.glob(args.input_file)
  for input_file in inputs:
      if args.height == None or args.width == None or args.bands == None:
          bands, width, height, endianess, datatype = get_geometry(input_file.split('/')[-1])
          geometry = '.'+input_file.split('/')[-1].split('.')[-2]+'.raw'
          path = input_file.replace(geometry,'')
          y = read_raw(input_file, int(height), int(width), int(bands), int(endianess), DTYPE=tf.float32)
      else:
          y = read_raw(input_file, args.height, args.width, args.bands, args.endianess, DTYPE=tf.float32)
          path = input_file
      x_hat = model.decompress1D(y)
      
      # Write reconstructed image out as a RAW file.
      write_raw(path+'_reverse-spectral-transform.'+bands+'_'+width+'_'+height+'_2_1_0.raw', x_hat)

def store_latent(args):
  """Compresses an image and stores the latent representation."""
  # Load model and use it to compress the image and save the latent representation.
  model = keras.models.load_model(args.model_path)
  inputs = glob.glob(args.input_file)
  for input_file in inputs:
      if args.height == None or args.width == None or args.bands == None:
          bands, width, height, endianess, datatype = get_geometry(input_file.split('/')[-1])
          geometry = '.'+input_file.split('/')[-1].split('.')[-2]+'.raw'
          path = input_file.replace(geometry,'')
          x = read_raw(input_file,int(height),int(width),int(bands),int(endianess))
      else:
          x = read_raw(input_file,args.height,args.width,args.bands,args.endianess)
          path = input_file
          
      y, x_shape, y_shape = model.main_transform(x)
      
      write_raw(path+'_latent-'+str(args.keep_latent)+'.'+str(int(tf.shape(y)[3]))+'_'+str(int(tf.shape(y)[1]))+'_'+str(int(tf.shape(y)[2]))+'_6_0_0.raw', y[args.keep_latent,:,:,:])

def transform(args):
  """Applies the main 1D-2D transform to an image."""
  # Load model and use it to compress the image.
  model = tf.keras.models.load_model(args.model_path)
  inputs = glob.glob(args.input_file)
  for input_file in inputs:
      if args.height == None or args.width == None or args.bands == None:
          bands, width, height, endianess, datatype = get_geometry(input_file.split('/')[-1])
          geometry = '.'+input_file.split('/')[-1].split('.')[-2]+'.raw'
          path = input_file.replace(geometry,'')
          x = read_raw(input_file, int(height), int(width), int(bands), int(endianess), data_type)
      else:
          x = read_raw(input_file, args.height, args.width, args.bands, args.endianess, data_type)
          path = input_file
      y, x_shape, y_shape = model.main_transform(x)
      
      y_3D = tf.transpose(y, (0,3,1,2))
      y_3D = tf.reshape(y_3D,(1, int(tf.shape(y)[-1])*int(bands), x_shape[0]//16, x_shape[1]//16))
      y_3D = tf.transpose(y_3D,(0,2,3,1))
      
      write_raw(path+'_noquant.'+str(int(tf.shape(y)[-1])*int(bands))+'_'+str(int(width)//16)+'_'+str(int(height)//16)+'_6_1_0.raw', y_3D[0,:,:,:])

def modulation(args):
  """Applies plain modulation to obtain the modulation vector from a lambda value."""
  # Load model and use it to compress the image.
  model = tf.keras.models.load_model(args.model_path)
  mod = model.plain_modulation(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.constant([args.quality]), 0), 0), 0))
  M = np.array(mod)
  M.tofile('./modulated_'+str(args.quality)+'.'+str(int(tf.shape(mod)[-1]))+'_1_1_6_1_0.raw', format='.raw')

def inverse_transform(args):
  """Applies the inverse 2D-1D transforms to a latent representation."""
  # Load model and use it to compress the image.
  model = tf.keras.models.load_model(args.model_path)
  inputs = glob.glob(args.input_file)
  for input_file in inputs:
      if args.height == None or args.width == None or args.bands == None:
          bands, width, height, endianess, datatype = get_geometry(input_file.split('/')[-1])
          geometry = '.'+input_file.split('/')[-1].split('.')[-2]+'.raw'
          path = input_file.replace(geometry,'')
          if datatype == '0':
            D = np.bool_
          elif datatype == '1':
            D = tf.uint8
          elif datatype == '2':
            D = tf.uint16
          elif datatype == '3':
            D = tf.int16
          elif datatype == '4':
            D = tf.int32
          elif datatype == '5':
            D = tf.int64
          elif datatype == '6':
            D = tf.float32
          else:
            D = tf.float64
          
          y = read_raw(input_file, int(height), int(width), int(bands), int(endianess), D)
      else:
          y = read_raw(input_file, args.height, args.width, args.bands, args.endianess, tf.float32)
          path = input_file
          bands, width, height, endianess = args.bands, args.width, args.height, args.endianess
      
      x_hat = model.inverse_transform(tf.expand_dims(y, 0), tf.constant([16*int(height), 16*int(width)]))
      
      write_raw(path+'.'+str(int(tf.shape(x_hat)[-1]))+'_'+str(int(width)*16)+'_'+str(int(height)*16)+'_2_1_0.raw.tfci.raw', x_hat[0,:,:,:])

def generate_coefficients(args):
    """Generates the regression coefficients of a model for a test set and stores them in a file."""
    from sklearn.linear_model import LinearRegression
    
    # Load model
    model = tf.keras.models.load_model(args.model_path)
    
    # For every quality parameter, calculate modulation magnitude
    qualities = []
    magnitudes = []
    for Qbin in range(256):
        quality = args.minimum_quality+(Qbin*((args.maximum_quality-args.minimum_quality)/256))
        mod = model.plain_modulation(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.constant([quality]), 0), 0), 0))
        mod_magnitude = float(tf.reduce_mean(mod))
        qualities.append([quality])
        magnitudes.append(mod_magnitude)
    qualities = np.array(qualities)
    magnitudes = np.array(magnitudes)
    
    # Linear regression for modulator
    regressor = LinearRegression().fit(qualities, magnitudes)
    b = float(regressor.predict(np.array([[0]])))
    a = float(regressor.predict(np.array([[1]])))-b
    
    # Compression and decompression at various qualities
    inputs = glob.glob(args.input_file)
    for input_file in inputs:
        if args.height == None or args.width == None or args.bands == None:
            bands, width, height, endianess, datatype = get_geometry(input_file.split('/')[-1])
            x = read_raw(input_file, int(height), int(width), int(bands), int(endianess), data_type)
        else:
            x = read_raw(input_file, args.height, args.width, args.bands, args.endianess, data_type)
            bands, width, height, endianess = args.bands, args.width, args.height, args.endianess
        
        y, x_shape, y_shape = model.main_transform(x)
        
        y_3D = tf.transpose(y, (0,3,1,2))
        y_3D = tf.reshape(y_3D,(1, int(tf.shape(y)[-1])*int(bands), x_shape[0]//16, x_shape[1]//16))
        y_3D = tf.transpose(y_3D,(0,2,3,1))
                
        # MSE baseline calculation
        x_hat_0 = tf.cast(tf.cast(model.inverse_transform(y_3D, tf.constant([int(height), int(width)])), tf.int32), tf.float32)
        x = tf.cast(tf.cast(x, tf.int32), tf.float32)
        mse_0 = float(tf.reduce_mean((x-x_hat_0)**2))
        
        mpqq_mmqq = 0
        mmqqqq = 0
        
        for Qbin in range(0, 256, 32):
            quality = args.minimum_quality+(Qbin*((args.maximum_quality-args.minimum_quality)/256))
            Q = tf.ones((1, 1, 1, 1))*quality
            
            string, side_string, z_shape, L, bps = model.modulation(y, x_shape, y_shape, Q)
            tensors = string, side_string, x_shape, y_shape, z_shape, L
            x_hat = tf.cast(tf.cast(model.decompress(*tensors), tf.int32), tf.float32)
            
            mse = float(tf.reduce_mean((x-x_hat)**2))
            quant_step = 1/(2*((a*quality)+b))
            mpqq_mmqq += mse_0*(mse-mse_0)*(quant_step**2)
            mmqqqq += (mse_0**2)*(quant_step**4)
    
    # Calculate quadratic regression coefficient
    alpha = mpqq_mmqq/mmqqqq
    
    # Write coefficients in file
    f = open(args.model_path+'/fixed_quality_coefficients.config', 'w')
    f.write(str(alpha)+','+str(a)+','+str(b))
    f.close()

def FQcompress(args):
  """Compresses an image."""
  import skimage.measure
  # Load model and use it to compress the image.
  model = tf.keras.models.load_model(args.model_path)
  inputs = glob.glob(args.input_file)
  for input_file in inputs:
      if args.height == None or args.width == None or args.bands == None:
          bands, width, height, endianess, datatype = get_geometry(input_file.split('/')[-1])
          x = read_raw(input_file, int(height), int(width), int(bands), int(endianess), data_type)
      else:
          x = read_raw(input_file, args.height, args.width, args.bands, args.endianess, data_type)
          bands, width, height, endianess = args.bands, args.width, args.height, args.endianess
      y, x_shape, y_shape = model.main_transform(x)
      
      y_3D = tf.transpose(y, (0,3,1,2))
      y_3D = tf.reshape(y_3D,(1, int(tf.shape(y)[-1])*int(bands), x_shape[0]//16, x_shape[1]//16))
      y_3D = tf.transpose(y_3D,(0,2,3,1))
      
      # MSE baseline calculation
      x_hat_0 = tf.cast(tf.cast(model.inverse_transform(y_3D, tf.constant([int(height), int(width)])), tf.int32), tf.float32)
      x = tf.cast(tf.cast(x, tf.int32), tf.float32)
      mse_0_full = (x-x_hat_0)**2
      mse_0_full_np = np.array(mse_0_full)
      mse_0 = skimage.measure.block_reduce(mse_0_full_np, (1, args.patchsize, args.patchsize, 1), np.mean)
      
      # Load regression coefficients
      f = open(args.model_path+'/fixed_quality_coefficients.config', 'r')
      coefs = f.read().split(',')
      alpha = float(coefs[0])
      a = float(coefs[1])
      b = float(coefs[2])
      
      # Quality parameters prediction
      target_magnitude = np.sqrt((alpha*mse_0)/(4*np.maximum(args.target_MSE-mse_0, 0.0001)))
      target_quality = np.mean(((target_magnitude-b)/a).astype(np.float32), axis = -1, keepdims=True)
      Q = target_quality.repeat(args.patchsize//16, axis=1).repeat(args.patchsize//16, axis=2)
            
      string_max, side_string_max, z_shape, L_max, bps_max = model.modulation(y, x_shape, y_shape, Q)
      
      string, side_string, L = string_max, side_string_max, L_max
      
      L = tf.reshape(L, (-1,))
      tensors = string, side_string, x_shape, y_shape, z_shape, L
    
      # Write a binary file with the shape information and the compressed string.
      packed = tfc.PackedTensors()
      packed.pack(tensors)
      with open(input_file+'.tfci', "wb") as f:
        f.write(packed.string)


def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument(
      "--verbose", "-V", action="store_true",
      help="Report progress and metrics when training or compressing.")
  parser.add_argument(
      "--model_path", default="CSMR2023061401_v4_0",
      help="Path where to save/load the trained model.")
  parser.add_argument(
      "--CPU", action="store_true",
      help="Mark to force the model to only run on CPU and temporarily disable GPU devices."
      "This is necessary in some hyperprior architectures for testing.")
  subparsers = parser.add_subparsers(
      title="commands", dest="command",
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'compress' reads an image file (lossless "
           "RAW format) and writes a compressed binary file. 'decompress' "
           "reads a binary file and reconstructs the image (in RAW format). "
           "input and output filenames need to be provided for the latter "
           "two options. Invoke '<command> -h' for more information.")

  # 'train' subcommand.
  train_cmd = subparsers.add_parser(
      "train",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Trains (or continues to train) a new model. Note that this "
                  "model trains on a continuous stream of patches drawn from "
                  "the training image dataset. An epoch is always defined as "
                  "the same number of batches given by --steps_per_epoch. "
                  "The purpose of validation is mostly to evaluate the "
                  "rate-distortion performance of the model using actual "
                  "quantization rather than the differentiable proxy loss. "
                  "Note that when using custom training images, the validation "
                  "set is simply a random sampling of patches from the "
                  "training set.")
  train_cmd.add_argument(
      "--lambda", type=float, nargs=4, default=[32737, 131072, 50, 0.000000001], dest="lmbda",
      help="Lambda that regulates (1 & 2) the minimum and macimum weight of the 2D mse in the loss function, (3) the orthogonal regularisation term, and (4) the variance for the 1D transform.")
  train_cmd.add_argument(
      "--train_glob", type=str, default=None,
      help="Glob pattern identifying custom training data. This pattern must "
           "expand to a list of RGB images in raw format. If unspecified, the "
           "CLIC dataset from TensorFlow Datasets is used.")
  train_cmd.add_argument(
      "--num_filters", type=int, nargs=4, default=[64,192,64,192],
      help="Number of filters per layer. The first input will be the number of "
      "filters in the hidden layers of the main (2D) transform and the second "
      "input in the latent layers of the main (2D) transform. Third and fourth "
      "inputs are the number of hidden and latent filters in the hyperprior.")
  train_cmd.add_argument(
      "--num_scales", type=int, default=64,
      help="Number of Gaussian scales to prepare range coding tables for.")
  train_cmd.add_argument(
      "--scale_min", type=float, default=.11,
      help="Minimum value of standard deviation of Gaussians.")
  train_cmd.add_argument(
      "--scale_max", type=float, default=256.,
      help="Maximum value of standard deviation of Gaussians.")
  train_cmd.add_argument(
      "--train_path", default="/tmp/CSMR2023061401_v4_0",
      help="Path where to log training metrics for TensorBoard and back up "
           "intermediate model checkpoints.")
  train_cmd.add_argument(
      "--batchsize", type=int, default=8,
      help="Batch size for training and validation.")
  train_cmd.add_argument(
      "--patchsize", type=int, default=256,
      help="Size of image patches for training and validation.")
  train_cmd.add_argument(
      "--epochs", type=int, default=100,
      help="Train up to this number of epochs. (One epoch is here defined as "
           "the number of steps given by --steps_per_epoch, not iterations "
           "over the full training dataset.)")
  train_cmd.add_argument(
      "--steps_per_epoch", type=int, default=1000,
      help="Perform validation and produce logs after this many batches.")
  train_cmd.add_argument(
      "--max_validation_steps", type=int, default=16,
      help="Maximum number of batches to use for validation. If -1, use one "
           "patch from each image in the training set.")
  train_cmd.add_argument(
      "--preprocess_threads", type=int, default=16,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")
  train_cmd.add_argument(
      "--check_numerics", action="store_true",
      help="Enable TF support for catching NaN and Inf in tensors.")
  train_cmd.add_argument(
      "--bands", type=int, default=3, dest="bands",
      help="Number of bands in the images to train the model.")
  train_cmd.add_argument(
      "--width", type=int, default=256, dest="width",
      help="Width of the images to train the model. All must be the same size.")
  train_cmd.add_argument(
      "--height", type=int, default=256, dest="height",
      help="Height of the images to train the model. All must be the same size.")
  train_cmd.add_argument(
      "--endianess", type=int, default=1, dest="endianess",
      help="Set to 0 if data is big endian, 1 if it's little endian.")
  train_cmd.add_argument(
      "--learning_rate", type=float, default=1e-4, dest="learning_rate",
      help="Learning rate for the training session.")
  train_cmd.add_argument(
      "--initialisation", type=str, default=None,
      help="Path to a RAW file containing the weights for spectral transform initialisation. These are expected to be stored as float32, and to be in the shape (bands, bands).")
  train_cmd.add_argument(
      "--progressive", type=int, nargs=3, default=[0,1,1], dest="progressive",
      help="Parameters for progressive sampling. First entry corresponds to the minimum band that may be sampled. Second entry is the maximum band index that may be sampled. Third and last entry is the stepsize of the sampling (bin size). Keep in mind the band indexes start at 0.")


  # 'compress' subcommand.
  compress_cmd = subparsers.add_parser(
      "compress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a raw file, compresses it, and writes a TFCI file.")

  # 'decompress' subcommand.
  decompress_cmd = subparsers.add_parser(
      "decompress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a TFCI file, reconstructs the image, and writes back "
                  "a raw file.")
  
  # 'transform' subcommand.
  transform_cmd = subparsers.add_parser(
      "transform",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a raw file, applies the main transform on it, and writes a raw file.")
  
  # 'inverse transform' subcommand.
  inv_transform_cmd = subparsers.add_parser(
      "inverse_transform",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a raw file float32, applies the decoder main transform on it, and writes a raw file.")
  
  # 'transform' subcommand.
  modulation_cmd = subparsers.add_parser(
      "modulation",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Given a quality parameter (lambda), constructs the modulation vector and stores it as a raw file.")
  
  # 'compress1D' subcommand.
  compress1D_cmd = subparsers.add_parser(
      "compress1D",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a raw file, applies the learned spectral transform to it, and writes a TFCI file in float32.")

  # 'decompress' subcommand.
  decompress1D_cmd = subparsers.add_parser(
      "decompress1D",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a RAW file, compresses the image only spectrally, then reconstructs the image only spectrally, and writes back "
                  "a raw file.")
  
   # 'generate coefficients' subcommand.
  generate_cmd = subparsers.add_parser(
      "generate_coefficients",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Generates regression coefficients for a model on a test set and stores them in the model folder.")
  
  # 'fixed-quality compress' subcommand.
  FQcompress_cmd = subparsers.add_parser(
      "FQcompress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a raw file, compresses it at a user-defined fixed MSE in some user-defined patches, and writes a TFCI file.")

  
  compress_cmd.add_argument("--quality",
                            type=float, default=0.1,
                            help="Lambda value regulating the rate-distortion tradeoff. A higher lambda should yield higher quality and rate images. If --bitrate is used, this lambda value will be used as the maximum lambda value in binary search."
                            )
  compress_cmd.add_argument("--quality_array",
                            type=str,
                            help="Array with lambda values regulating the rate-distortion tradeoff for each latent pixel."
                            )
  compress_cmd.add_argument("--bitrate",
                            type=float, default=None,
                            help="Bitrate to compress the images at, in bits per sample (bps, equivalent to bits per pixel per band)."
                            )
  modulation_cmd.add_argument("quality",
                            type=float, default=0.0001,
                            help="Lambda value regulating the rate-distortion tradeoff."
                            )
  generate_cmd.add_argument("--minimum_quality",
                            type=float, default=0.0001,
                            help="Minimum quality used in the model to generate coefficients."
                            )
  generate_cmd.add_argument("--maximum_quality",
                            type=float, default=0.01,
                            help="Maximum quality used in the model to generate coefficients."
                            )
  
  # Arguments for both 'compress' and 'decompress'.
  for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".raw"), (compress1D_cmd, ".tfci"), (decompress1D_cmd, ".raw"), (transform_cmd, ".tfci"), (inv_transform_cmd, ".raw"), (generate_cmd, ".raw"), (FQcompress_cmd, ".raw")):
    cmd.add_argument(
        "input_file",
        help='Input filename or glob pattern. If a glob pattern is used, delimitate it with "".')
    cmd.add_argument(
        "--bands",type=int,default=None,
        help="Number of bands in input image.")
    cmd.add_argument(
        "--width",type=int,default=None,
        help="Input image width.")
    cmd.add_argument(
        "--height",type=int,default=None,
        help="Input image height.")
    cmd.add_argument(
        "--endianess",type=int,default=None,
        help="Set to 0 if data is big endian, 1 if it's little endian.")
  
  compress_cmd.add_argument(
          "--keep_latent", type=int, default=None,
          help="When selected, stores the latent representation of the image as it is produced by the main transform. Indicate the number of frame you wish to store.")
  
  FQcompress_cmd.add_argument("target_MSE",
                            type=float,
                            help="Target MSE for fixed-quality compression. Will be homogeneous for all blocks."
                            )
  FQcompress_cmd.add_argument("patchsize",
                            type=int,
                            help="Size of blocks where to aim for fixed-quality compression."
                            )

  
  # Parse arguments.
  args = parser.parse_args(argv[1:])
  if args.command is None:
    parser.print_usage()
    sys.exit(2)
  return args


def main(args):
  # Invoke subcommand.
  if args.CPU:
      os.environ["CUDA_VISIBLE_DEVICES"]="-1"
  if args.command == "train":
    train(args)
  elif args.command == "compress":
    compress(args)
    if not args.keep_latent==None:
        store_latent(args)
  elif args.command == "decompress":
    decompress(args)
  elif args.command == "compress1D":
    compress1D(args)
  elif args.command == "decompress1D":
    decompress1D(args)
  elif args.command == "transform":
    transform(args)
  elif args.command == "inverse_transform":
    inverse_transform(args)
  elif args.command == "modulation":
    modulation(args)
  elif args.command == "generate_coefficients":
    generate_coefficients(args)
  elif args.command == "FQcompress":
    FQcompress(args)



if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)
