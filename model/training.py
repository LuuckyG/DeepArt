import tensorflow as tf

from .model import StyleModel

# Content layer where will pull our feature maps
content_layers = ['block5_conv2']

# Style layer of interest
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

style_model = StyleModel(style_layers=style_layers, content_layers=content_layers)
vgg19 = style_model.vgg
