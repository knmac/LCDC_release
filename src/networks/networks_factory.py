from __future__ import absolute_import
from __future__ import print_function

import sys

# from .VGGnet_train import VGGnetTrain
from .VGG_vanilla_opticalflow_v2 import VGGVanillaOpticalFlowV2
# from .Resnet50_train import Resnet50Train
from .Resnet50_vanilla import Resnet50Vanilla
from .Resnet50_vanilla_plus import Resnet50VanillaPlus
from .Resnet50_vanilla_plusplus import Resnet50VanillaPlusPlus
from .Resnet50_vanilla_opticalflow import Resnet50VanillaOpticalFlow
from .Resnet50_encoder import Resnet50Encoder  # DC
from .Resnet50_motionconstraint import Resnet50_MotionConstraint  # LCDC


def build_net(netname, n_classes, snippet_len, height, width, max_time_gap=1,
              trainable=True):
    """ build network according to given netname

    Args:
        netname: name of the network
        n_classes: number of classes
        snippet_len: number of frames per snippet
        height: height of a frame
        width: width of a frame
        max_time_gap: maximum of time widening to apply. Must be at least
            1 and less than snippet_len
        trainable: whether the network is trainable

    Returns:
        network with the given netname
    """
    net_dict = {
        # 'vgg16': VGGnetTrain,
        'vgg_vanilla_opticalflow_v2': VGGVanillaOpticalFlowV2,
        'resnet50_vanilla': Resnet50Vanilla,
        'resnet50_vanilla_plus': Resnet50VanillaPlus,
        'resnet50_vanilla_plusplus': Resnet50VanillaPlusPlus,
        'resnet50_vanilla_opticalflow': Resnet50VanillaOpticalFlow,
        # 'resnet50': Resnet50Train,
        'resnet50_encoder': Resnet50Encoder,
        'resnet50_motionconstraint': Resnet50_MotionConstraint,
    }

    assert netname in net_dict, \
        'Network name not supported: {}'.format(netname)
    return net_dict[netname](n_classes, snippet_len, height, width, max_time_gap, trainable)
