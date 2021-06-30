"""General-purpose test script for synthetic image generation using CycleGAN.

Once you have trained your model with train.py, you can use this script to generate images.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It will generate images from domain A to B or B to A, according to '--direction'
It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to the disk.


See options/base_options.py and options/test_options.py for more test options.
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util
import ntpath

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.phase = 'train'   # set phase to train to create the train dataset
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    opt.phase = 'test'    # set phase to test to set the model to test mode
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        gen_img = model.get_current_visuals()['fake_B']  # get image results
        
        img_path = model.get_image_paths()     # get original image paths
        short_path = ntpath.basename(img_path[0])
        name = os.path.splitext(short_path)[0]
        
         # save to the disk
        if not os.path.exists(opt.results_dir):
            util.mkdirs(opt.results_dir)
        save_img = util.tensor2im(gen_img)
        image_name = '%s_gen.png' % (name)
        save_path = os.path.join(opt.results_dir, image_name)
        util.save_image(save_img, save_path, aspect_ratio=opt.aspect_ratio)
        
        if i % 100 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
