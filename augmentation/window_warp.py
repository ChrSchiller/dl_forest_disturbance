import numpy as np

### modified from https://github.com/uchidalab/time_series_augmentation/blob/master/utils/augmentation.py
### modifications mainly necessary due to conversion to Pytorch from Keras/Tensorflow
### as we have only 1 sample here, while the original implementation considers a whole batch at once;
### but also because of use case in this study
def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales)
    warp_size = np.ceil(window_ratio * x.shape[0]).astype(int)
    window_steps = np.arange(warp_size)

    # print(1 > (x.shape[0] - warp_size - 1))
    if (x.shape[0] - warp_size - 1) > 10:
        window_starts = np.random.randint(low=1, high=x.shape[0] - warp_size - 1, size=1).astype(int)
        window_ends = (window_starts + warp_size).astype(int)

        ret = np.zeros_like(x)
        for dim in range(x.shape[1]):
            pat = x[:, dim]
            start_seg = pat[:window_starts[0]]
            window_seg = np.interp(np.linspace(0, warp_size - 1, num=int(warp_size * warp_scales)), window_steps,
                                   pat[window_starts[0]:window_ends[0]])
            end_seg = pat[window_ends[0]:]
            warped = np.concatenate((start_seg, window_seg, end_seg))
            ret[:, dim] = np.interp(np.arange(x.shape[0]), np.linspace(0, x.shape[0] - 1., num=warped.size),
                                       warped).T
        ret[:, -1] = ret[:, -1].astype(int)

        return ret
    else:
        return x