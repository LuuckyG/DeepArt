import argparse
from pathlib import Path


class HParams:
    """
    Class that takes in all model training parameters.
    """

    def __init__(self):
        self.args = self.get_parser()

    def get_parser(self):
        arg_parser = argparse.ArgumentParser()

        # Required arguments
        arg_parser.add_argument('--results_dir', '--rp', type=Path,
                                default="./src/results",
                                help="Folder where the model results are saved in subfolders.")
        arg_parser.add_argument('--content_path', '--cp',type=Path,
                                default="C:/Users/luukg/Downloads/Tim_en_Luuk.jpeg",
                                help="Folder where the model results are saved.")
        arg_parser.add_argument('--style_path', '--sp', type=Path,
                                default="./src/images/wave.jpg",
                                help="Folder where the model results are saved.")
                        
        # Training arguments
        arg_parser.add_argument('--content_layers', '--cl', nargs='+', type=str,
                                default='block4_conv2', help="Content layer where we will pull our feature maps from")
        arg_parser.add_argument('--style_layers', '--sl', nargs='+', type=str,
                                default=['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1','block5_conv1'], 
                                help="Style layer of interest")   
        arg_parser.add_argument('--num_iterations', '--ni', type=int,
                                default=1000, help="Number of training interations")
        arg_parser.add_argument('--display_num', '--dn', type=int,
                                default=100, help="After 'display_num' iterations an intermediate image is saved.")
        arg_parser.add_argument('--content_weight', '--cw', type=float,
                                default=1e3, help="Contribution of content loss to total loss")
        arg_parser.add_argument('--style_weight', '--sw', type=float,
                                default=1e-2, help="Contribution of style loss to total loss")
        arg_parser.add_argument('--learning_rate', '--lr', type=float,
                                default=5.0, help="Learning rate of Adam optimizer.")
        arg_parser.add_argument('--beta_1', '--b1', type=float,
                                default=0.99, help="Beta1 of Adam optimizer.")
        arg_parser.add_argument('--epsilon', '--ep', type=float,
                                default=1e-1, help="Epsilon of Adam optimizer.")

        return arg_parser.parse_args()
