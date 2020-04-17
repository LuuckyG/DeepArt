import time
import numpy as np
from PIL import Image
from pathlib import Path

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from .utils import load_img, deprocess_img, load_and_process_img
from ..hparams import HParams

# TensorFlow GPU settings
config = ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = InteractiveSession(config=config)


def gram_matrix(input_tensor):
    # We make the image channels first
    channels = int(input_tensor.shape[-1])
    input_tensor = tf.reshape(input_tensor, [-1, channels])
    num_locations = tf.shape(input_tensor)[0]
    gram = tf.matmul(input_tensor, input_tensor, transpose_a=True)
    return gram / tf.cast(num_locations, tf.float32)


class StyleModel:

    def __init__(self, results_path, content_path, style_path,
                 content_layers=('block1_conv1', 
                                 'block2_conv1', 
                                 'block3_conv1', 
                                 'block4_conv1',
                                 'block5_conv1'),
                 style_layers=('block4_conv2'),
                 num_iterations=1000,
                 content_weight=1e3, 
                 style_weight=1e-2,
                 display_num=100, 
                 learning_rate=5, 
                 beta_1=0.99, 
                 epsilon=1e-1):

        # Get training variables
        self.results_path = results_path
        self.content_path = content_path
        self.style_path = style_path
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_content_layers = len(self.content_layers)
        self.num_style_layers = len(self.style_layers)
        self.num_iterations = num_iterations
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.display_num = display_num
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.epsilon = epsilon

        # Get style model
        self.model = self.get_model()
        for layer in self.model.layers:
            layer.trainable = False

        # Get the style and content feature representations
        self.style_features, self.content_features = self.get_feature_representations()
        self.gram_style_features = [gram_matrix(style_feature) for style_feature in self.style_features]

        # Set initial image
        self.init_image = load_and_process_img(self.content_path)
        self.init_image = tf.Variable(self.init_image, dtype=tf.float32)

        # Create our optimizer
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, 
                                            beta_1=beta_1, 
                                            epsilon=epsilon)

        # Create a nice config 
        self.loss_weights = (style_weight, content_weight)
        self.cfg = {
            'model': self.model,
            'loss_weights': self.loss_weights,
            'init_image': self.init_image,
            'gram_style_features': self.gram_style_features,
            'content_features': self.content_features
        }
    
    def get_model(self):
        """ Creates our model with access to intermediate layers. 

        This function will load the VGG19 model and access the intermediate layers. 
        These layers will then be used to create a new model that will take input image
        and return the outputs from these intermediate layers from the VGG model. 

        Returns:
        returns a keras model that takes image inputs and outputs the style and 
            content intermediate layers. 
        """

        # Load our model. We load pretrained VGG, trained on imagenet data
        vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        # Get output layers corresponding to style and content layers 
        style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        content_outputs = [vgg.get_layer(name).output for name in self.content_layers]
        model_outputs = style_outputs + content_outputs

        # Build model 
        return tf.keras.models.Model(vgg.input, model_outputs)

    def get_feature_representations(self):
        """Helper function to compute our content and style feature representations.

        This function will simply load and preprocess both the content and style 
        images from their path. Then it will feed them through the network to obtain
        the outputs of the intermediate layers. 

        Arguments:
        model: The model that we are using.
        content_path: The path to the content image.
        style_path: The path to the style image

        Returns:
        returns the style features and the content features. 
        """
        # Load our images in 
        content_image = load_and_process_img(self.content_path)
        style_image = load_and_process_img(self.style_path)

        # batch compute content and style features
        style_outputs = self.model(style_image)
        content_outputs = self.model(content_image)

        # Get the style and content feature representations from our model  
        style_features = [style_layer[0] for style_layer in style_outputs[:self.num_style_layers]]
        content_features = [content_layer[0] for content_layer in content_outputs[self.num_style_layers:]]
        return style_features, content_features
    
    def train(self):
        start_time = time.time()
        global_start = time.time()

        norm_means = np.array([103.939, 116.779, 123.68])
        min_vals = -norm_means
        max_vals = 255 - norm_means

        # Store our best result
        best_loss, best_img = float('inf'), None

        for i in range(self.num_iterations):
            grads, all_loss = self.compute_grads(self.cfg)
            loss, style_score, content_score = all_loss
            # grads, _ = tf.clip_by_global_norm(grads, 5.0)
            self.opt.apply_gradients([(grads, self.init_image)])
            clipped = tf.clip_by_value(self.init_image, min_vals, max_vals)
            self.init_image.assign(clipped)

            if loss < best_loss:
                # Update best loss and best image from total loss. 
                best_loss = loss
                best_img = self.init_image.numpy()

            if i % self.display_num == 0:
                print('Iteration: {}'.format(i))        
                print('Total loss: {:.4e}, ' 
                      'style loss: {:.4e}, '
                      'content loss: {:.4e}, '
                      'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
                
                start_time = time.time()

                im = Image.fromarray(deprocess_img(self.init_image.numpy()))
                im.save(self.results_path.as_posix() + '/epoch_{}.jpg'.format(i))

        print('Total time: {:.4f}s'.format(time.time() - global_start))
            
        return best_img, best_loss