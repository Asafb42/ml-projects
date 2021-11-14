set -ex
python train.py --model cycle_gan --dirA Normal --dirB COVID --netG unet_256 --name COVID-19_Radiography_normal_covid_cyclegan --dataroot ../Datasets/COVID-19_Radiography_Dataset_organized --display_freq 5000 --print_freq 1000 --pool_size 50 --no_dropout --checkpoints_dir ../Models/COVID-19_Radiography_cyclegan/Checkpoints --lambda_A 5 --lambda_B 5 --continue_train --epoch_count 53