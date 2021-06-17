"""General-purpose evaluation script for classification taks.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

See options/base_options.py and options/test_options.py for more test options.
"""
import os
import numpy as np
import torch
import time
from util.measurments import analyze_results
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html


if __name__ == '__main__':
    torch.manual_seed(0)    # set a manual seed for evaluation reproducibility.
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    # opt.num_threads = 0   # test code only supports num_threads = 0
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display;
    opt.preprocess = 'resize'
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    predictions = []
    labels = []

    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()

    start_time = time.time()  # timer for computation.

    with torch.no_grad():
        for i, data in enumerate(dataset):
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break

            model.set_input(data)  # unpack data from data loader
            current_preds, current_labels = model.get_predictions() # forward data and calculate predictions        
            
            predictions.extend(current_preds.tolist())
            labels.extend(current_labels.tolist())

    end_time = time.time()  # timer for computation.
    
    predictions = np.array(predictions)
    labels = np.array(labels)

    # If it's not a binary classification problem calculate only accuracy. 
    # If it's a binary classification problem calculate numerus binary evalutaion metrics. 
    if opt.label_num > 2:
        test_acc = 100 * np.sum(predictions == labels) / len(dataset)
        print("Test evaluation results:\nTest size: %d\nAccuracy: %f\n" % (len(dataset), test_acc))
    else:
        auc, dice, ppv, sens, acc , npv, spec, tp, fn, fp, tn = analyze_results(predictions, labels)
        print("Test evaluation results:\nTest size: %d\nAUC Score: %f\nAccuracy: %f\nSensitivity: %f\nSpecificity: %f\nPrecision: %f\n" % (len(dataset), auc, acc, sens, spec, ppv))
    print("Computation time: %.2f sec" %(end_time - start_time))
