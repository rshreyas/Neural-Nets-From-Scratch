import numpy as np
import skimage
from pprint import pprint
import skimage.data
from skimage.viewer import ImageViewer

class Conv2D():
    def __init__(self, num_kernels=4, kernel_shape=[3,3], padding='valid', step_size=1):
        self.num_kernels = num_kernels
        self.kernel_shape = kernel_shape
        self.padding = padding
        self.step_size = step_size


    def _process_input_image(self, to_grayscale=True):
        img=skimage.data.chelsea()
        if to_grayscale:
            img = skimage.color.rgb2gray(img)
        return img


    def _init_filters(self, standard_norm_init=True):
        num_kernels = self.num_kernels
        input_shape = self.kernel_shape
        layer_filters = None

        # Sanity Checks - Filters must have equal dimensions, and must be odd
        assert(input_shape[0] == input_shape[1])
        assert(input_shape[0] % 2 != 0)

        if not standard_norm_init:
            # Initialize filters to 0s
            layer_filters = np.zeros(num_kernels, input_shape[0], input_shape[1])
        else:
             layer_filters = np.random.standard_normal(size=(num_kernels,
                                                            input_shape[0],
                                                            input_shape[1]))

        self.layer_filters = layer_filters


    def _check_channel_dims(self, img, layer_filters, channel_first=False):
        if len(img.shape) > 2 or len(layer_filters.shape) > 3:
            # At this point, either image or filters has been judged to be non-default
            if channel_first:
                assert(len(img.shape[0]) == len(layer_filters.shape[0]))
            else:
                assert(len(img.shape[-1]) == len(layer_filters.shape[-1]))
        return True


    def _convolve(self, img, curr_filter):
        # Both img and curr_filter guarenteed to be 2D matrix
        filter_size = curr_filter.shape[0]
        result = np.zeros((img.shape))

        for row in np.uint16(np.arange(filter_size/2, img.shape[0] - filter_size/2 - 2)):
            for col in np.uint16(np.arange(filter_size/2, img.shape[1] - filter_size/2 - 2)):
                sub_img = img[row: row + filter_size, col: col + filter_size]
                conv_res = curr_filter * sub_img
                conv_sum = np.sum(conv_res)
                result[row][col] = conv_sum

        result = result[np.uint16(filter_size / 2): result.shape[0] - np.uint16(filter_size/2),
                        np.uint16(filter_size / 2): result.shape[1] - np.uint16(filter_size/2)]
        return result


    def get_layer_feature_maps(self, img_source=None, to_grayscale=True, channel_first=False):
        if img_source:
            img = self._process_input_image(img_source, to_grayscale)
        else:
            img = self._process_input_image(to_grayscale)

        self._init_filters()
        layer_filters = self.layer_filters
        if self._check_channel_dims(img, layer_filters):
            channel_idx = 0
            if channel_first:
                channel_idx = 1

            if self.padding == 'valid':
                feature_maps = np.zeros((img.shape[0 + channel_idx] - layer_filters.shape[1] + 1,
                                         img.shape[1 + channel_idx] - layer_filters.shape[1] + 1,
                                         layer_filters.shape[0]))

            # TODO: feature_maps for 'same' padding
            for filter_idx in range(layer_filters.shape[0]):
                curr_filter = layer_filters[filter_idx, :]

                if len(curr_filter.shape) == 2:
                    # There is only one channel
                    feature_map = self._convolve(img, curr_filter)
                else:
                    # Find the feature maps for each channel, and sum them pointwise
                    if channel_first:
                        feature_map = self._convolve(img[0, :, :], curr_filter[0, :, :])
                        for channel_idx in range(1, img.shape[0]):
                            feature_map += self._convolve(img[channel_idx, :, :],
                                                          curr_filter[channel_idx, :, :])
                    else:
                        feature_map = self._convolve(img[:, :, 0], curr_filter[:, :, 0])
                        for channel_idx in range(1, img.shape[-1]):
                            feature_map += self._convolve(img[:, :, channel_idx],
                                                          curr_filter[:, :, channel_idx])

                feature_maps[:, :, filter_idx] = feature_map
        return feature_maps


if __name__ == "__main__":
    c1 = Conv2D()
    #c1.input_image(to_grayscale=True)
    feature_maps = c1.get_layer_feature_maps()
    viewer = ImageViewer(feature_maps[:,:,0])
    viewer.show()
    viewer = ImageViewer(feature_maps[:,:,1])
    viewer.show()
    viewer = ImageViewer(feature_maps[:,:,2])
    viewer.show()
    viewer = ImageViewer(feature_maps[:,:,3])
    viewer.show()
