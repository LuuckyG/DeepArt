from PIL import Image
from pathlib import Path
from datetime import datetime

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from .hparams import HParams
from .model.utils import deprocess_img
from .model.style_model import StyleModel

# TensorFlow GPU settings
config = ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = InteractiveSession(config=config)


def main(args):

    # Create folder
    style_name = args.style_path.stem()
    style_name = style_name.split()[0]

    time_now = time_now = datetime.now()  

    results_path = Path(args.results_dir /
                    ('_i_' + style_name 
                    + '_sl_' + args.style_layers
                    + '_lr_' + str(args.learning_rate) 
                    + '_b_' + str(args.beta_1) 
                    + '_e_' + str(args.epsilon) 
                    + '_' + time_now.strftime('%H-%M')))
    results_path.mkdir(exist_ok=True, parents=True)

    style_model = StyleModel(results_path=results_path, 
                             content_path=args.content_path, 
                             style_path=args.style_path,
                             content_layers=args.content_layers,
                             style_layers=args.style_layers,
                             num_iterations=args.num_iterations,
                             content_weight=args.content_weight, 
                             style_weight=args.style_weight,
                             display_num=args.display_num, 
                             learning_rate=args.learning_rate, 
                             beta_1=args.beta_1, 
                             epsilon=args.epsilon)

    best, best_loss = style_model.train()

    # Save best result
    im = Image.fromarray(deprocess_img(best))
    im.save(results_path.as_posix() + '/best.jpg')


if __name__ == '__main__':
    parameters = HParams().args
    main(parameters)
