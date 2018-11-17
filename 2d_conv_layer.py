import numpy as np
import skimage
from pprint import pprint
import skimage.data
from skimage.viewer import ImageViewer
from skimage.transform import resize

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


    def _init_filters(self, standard_norm_init=True, num_channels=1):
        num_kernels = self.num_kernels
        input_shape = self.kernel_shape
        layer_filters = None

        # Sanity Checks - Filters must have equal dimensions, and must be odd
        assert(input_shape[0] == input_shape[1])
        assert(input_shape[0] % 2 != 0)

        if not standard_norm_init:
            # Initialize filters to 0s
            if num_channels > 1:
                layer_filters = np.zeros(num_kernels, input_shape[0], input_shape[1], num_channels)
            else:
                layer_filters = np.zeros(num_kernels, input_shape[0], input_shape[1])
        else:
            if num_channels > 1:
                layer_filters = np.random.standard_normal(size=(num_kernels,
                                                                input_shape[0],
                                                                input_shape[1],
                                                                num_channels))
            else:
                layer_filters = np.random.standard_normal(size=(num_kernels,
                                                                input_shape[0],
                                                                input_shape[1]))

        self.layer_filters = layer_filters


    def _check_channel_dims(self, img, layer_filters, channel_first=False):
        if len(img.shape) > 2 or len(layer_filters.shape) > 3:
            # At this point, either image or filters has been judged to be non-default
            if channel_first:
                assert(img.shape[0] == layer_filters.shape[0])
            else:
                assert(img.shape[-1] == layer_filters.shape[-1])
        return True


    def _convolve(self, img, curr_filter):
        # Both img and curr_filter guarenteed to be 2D matrix
        filter_size = curr_filter.shape[0]
        result = np.zeros((img.shape))

        #print(np.uint16(np.arange(filter_size/2, img.shape[0] - filter_size/2 - 2)))
        #print(np.uint16(np.arange(filter_size/2, img.shape[1] - filter_size/2 - 2)))

        for row in range(img.shape[0] - filter_size + 1):
            for col in range(img.shape[1] - filter_size + 1):
                sub_img = img[row: row + filter_size, col: col + filter_size]
                conv_res = curr_filter * sub_img
                conv_sum = np.sum(conv_res)
                result[row][col] = conv_sum

        result = result[: result.shape[0] - filter_size + 1,
                        : result.shape[1] - filter_size + 1]
        return result


    def get_layer_feature_maps(self, img_source=None, to_grayscale=True, channel_first=False):
        if img_source is not None:
            img = img_source
            self._init_filters(standard_norm_init=True, num_channels=img_source.shape[-1])
        else:
            img = self._process_input_image(to_grayscale)
            if to_grayscale:
                self._init_filters(standard_norm_init=True, num_channels=1)
            else:
                self._init_filters(standard_norm_init=True, num_channels=3)

        layer_filters = self.layer_filters

        if self._check_channel_dims(img, layer_filters):
            channel_idx = 0
            if channel_first:
                channel_idx = 1
            if self.padding == 'valid':
                feature_maps = np.zeros((img.shape[0 + channel_idx] - layer_filters.shape[1] + 1,
                                         img.shape[1 + channel_idx] - layer_filters.shape[1] + 1,
                                         self.num_kernels))

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


def relu(feature_maps):
    relu_out = np.zeros(feature_maps.shape)
    for feature_dim in range(feature_maps.shape[-1]):
        for row in range(feature_maps.shape[0]):
            for col in range(feature_maps.shape[1]):
                relu_out[row, col, feature_dim] = np.max(feature_maps[row, col, feature_dim], 0)
    return relu_out


def maxpool_2d(feature_maps, size=2, stride=2):
    maxpool_out = np.empty(((feature_maps.shape[0] - size + 1) / stride,
                            (feature_maps.shape[1] - size + 1) / stride,
                            feature_maps.shape[-1]))

    for map_idx in range(feature_maps.shape[-1]):
        row_pool_idx = 0
        for row in range(0, feature_maps.shape[0] - size - 1, stride):
            col_pool_idx = 0
            for col in range(0, feature_maps.shape[1] - size - 1, stride):
                maxpool_out[row_pool_idx, col_pool_idx, map_idx] = np.max(feature_maps[row: row + size, col: col + size, map_idx])
                col_pool_idx += 1
            row_pool_idx += 1
    return maxpool_out


def display_conv_relu_pool_block(feature_maps, relu_out, maxpool_out):
    img_list = []
    for i in range(feature_maps.shape[-1]):
        img = np.concatenate((feature_maps[:,:,i],
                              relu_out[:,:,i],
                              resize(maxpool_out[:,:,i],
                                     (relu_out.shape[0],
                                     relu_out.shape[1]))),
                              axis=1)
        img_list.append(img)
    img_final = np.concatenate(([i for i in img_list]), axis=0)

    viewer = ImageViewer(img_final)
    viewer.show()


if __name__ == "__main__":
    c1 = Conv2D()
    feature_maps_1 = c1.get_layer_feature_maps()
    relu_out_1 = relu(feature_maps_1)
    maxpool_out_1 = maxpool_2d(relu_out_1)
    display_conv_relu_pool_block(feature_maps_1, relu_out_1, maxpool_out_1)

    c2 = Conv2D()
    feature_maps_2 = c2.get_layer_feature_maps(img_source=maxpool_out_1)
    relu_out_2 = relu(feature_maps_2)
    maxpool_out_2 = maxpool_2d(relu_out_2)
    display_conv_relu_pool_block(feature_maps_2, relu_out_2, maxpool_out_2)
