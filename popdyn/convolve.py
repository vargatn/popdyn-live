import numpy as np


def distance_kernel(half_extent=50):
    size = half_extent * 2 + 1
    kernel = np.zeros((size, size))
    vals = np.arange(-half_extent, half_extent + 1)
    xx, yy = np.meshgrid(vals, vals, indexing='xy')
    # y, x = np.ogrid[-half_extent:half_extent+1, -half_extent:half_extent+1]
    distance = np.sqrt(xx ** 2 + yy ** 2)
    return distance


def tophat(dists, radius=50.):
    kernel = np.zeros(dists.shape)
    kernel[dists < radius] = 1
    kernel = kernel / kernel.sum()
    return kernel


def gaussian(dists, scale=50.):
    kernel = np.zeros(dists.shape)
    kernel = np.exp(- dists ** 2. / (2 * scale ** 2.))
    kernel = kernel / kernel.sum()
    return kernel


def triangle(dists, scale=50):
    """triangle kernel, scale is when it hits zero"""
    kernel = np.zeros(dists.shape)
    kernel = scale - dists
    kernel[kernel < 0] = 0
    kernel = kernel / kernel.sum()
    return kernel


kernels = {
    "gauss": gaussian,
    "tophat": tophat,
    "triangle": triangle,
}


def make_noise_map(scale, shape, which="gauss"):
    num = shape[0] * shape[1]
    noise = np.random.uniform(size=num).reshape(shape)
    dists = distance_kernel(half_extent=scale * 3)

    kern = kernels[which](dists, scale=scale)
    tmp = np.fft.irfft2(np.fft.rfft2(noise) * np.fft.rfft2(kern, noise.shape))
    tmp = tmp - tmp.min()
    tmp = tmp / tmp.max()

    return tmp


def convolve_map(canvas, scale, which="gauss", ):
    pad_width = scale * 3
    padded = np.pad(canvas, pad_width=scale * 3)
    half_extent = scale * 2
    dists = distance_kernel(half_extent=half_extent)
    kern = kernels[which](dists, scale=scale)

    # There's some shape difference here, which needs to be manually corrected with the padding
    # just offset the convolved image by the half_extent of the kernel...
    tmp = np.fft.irfft2(np.fft.rfft2(padded) * np.fft.rfft2(kern, padded.shape))
    tmp = tmp[pad_width + half_extent:-pad_width + half_extent, pad_width + half_extent:-pad_width + half_extent]
    return tmp