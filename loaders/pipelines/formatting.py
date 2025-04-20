# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Yang Chen
# ------------------------------------------------------------------------
import numpy as np
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
from mmdet3d.core.points import BasePoints
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets.pipelines import DefaultFormatBundle

@PIPELINES.register_module()
class SparseOccFormatBundle3D(DefaultFormatBundle):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - collect_keys: (1)to tensor, (2)to DataContainer
    """

    def __init__(self, class_names, collect_keys):
        super(SparseOccFormatBundle3D, self).__init__()
        self.class_names = class_names
        self.collect_keys = collect_keys
    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # NOTE add process for self.collect_keys
        for key in self.collect_keys:
            if key in ['timestamp',  'img_timestamp']:
                results[key] = DC(to_tensor(np.array(results[key], dtype=np.float64)))
            else:
                results[key] = DC(to_tensor(np.array(results[key], dtype=np.float32)))

        results = super(SparseOccFormatBundle3D, self).__call__(results)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(class_names={self.class_names}, '
        repr_str += f'collect_keys={self.collect_keys})'
        return repr_str