
# %%
import fibresegt as fs
import numpy as np
import matplotlib.pyplot as plt

undersample_factor = [8, 12]
noise_level = [4, 8, 12, 16, 20, 24]

# Datasets
dataset_folder = '/scratch/pooja/fiber_detection/data/final_data'
Mock_UD = '/ScanRef_Glass_Mock_UD/'
noisy_Mock = 'im_Mock500noisy'
mask_Mock = 'curatedmask_Mock500.png'

def divide_img_blocks(img, n_blocks=(2,2)):
   horizontal = np.array_split(img, n_blocks[0])
   splitted_img = [np.array_split(block, n_blocks[1], axis=1) for block in horizontal]
   return splitted_img

# %%

for i in range(len(undersample_factor)):
    for j in range(len(noise_level)):
        output_dir = './final_output_detect/Mock_2_noisy_uf' + str(undersample_factor[i]) + '_n' + str(noise_level[j]) + '/'
        im_Mock = noisy_Mock + '_uf' + str(undersample_factor[i]) + '_n' + str(noise_level[j]) + '.tiff'
        dataset_folder1 = dataset_folder + Mock_UD
        dataset_file = fs.join(dataset_folder1, im_Mock)
        origData = np.array(fs.imread(dataset_file))
        origData = fs.normalize_8_bit(origData)
        label_file = fs.join(dataset_folder1, mask_Mock)
        data_info = dict(dataset_file=dataset_file, 
                 label_file=label_file)
        labelInnerFibre = np.array(fs.imread(label_file))
        if labelInnerFibre.ndim > 2:
            labelInnerFibre = labelInnerFibre[:,:,0]
        image_size          = (64, 64)
        train_stride_step   = 8
        itera_enlarge_size  = 2
        itera_shrink_size  = 1
        fig_path            = fs.join(dataset_folder, 'crop_data')
        save_fig            = False
        sample_info         = dict(image_size = image_size, 
                                train_stride_step=train_stride_step) 
        divided_img = divide_img_blocks(origData, n_blocks=(2,2))
        divided_label = divide_img_blocks(labelInnerFibre, n_blocks=(2,2))

        train_data = divided_img[1][0]
        label_data = divided_label[1][0]

        Data = fs.generate_training_samples(data=[train_data, label_data], 
                                    stride_step=train_stride_step, 
                                    data_shape=image_size, 
                                    itera_enlarge_size=itera_enlarge_size,
                                    fig_path=fig_path,
                                    save_fig=save_fig,
                                    show_img=True)
        Data               = Data
        image_size         = (64, 64)
        val_percent        = 0.20 # The probability to split the data as training and testing
        epochs             = 100
        batch_size         = 16
        learning_rate      = 0.001
        net_var            = 'UnetID'
        # data_aug           = {'brightness':0.3, 'contrast':0.3,
        #                       'GaussianBlur_kernel':3, 'GaussianBlur_sigma': (0.7, 1.1)}
        data_aug           = None
        preprocess_info    = dict(data_info=data_info,
                                sample_info=sample_info)
        fs.apis.train_net(Data=Data, image_size=image_size, output_dir=output_dir, 
                  val_percent=val_percent, epochs=epochs, batch_size=batch_size,
                  learning_rate=learning_rate, net_var=net_var, 
                  data_aug=data_aug,
                  preprocess_info=preprocess_info)





# %%

# output_dir = './final_output_detect/Mock_1/'
# im_Mock = 'im_Mock500.tiff'
# dataset_folder1 = dataset_folder + Mock_UD
# dataset_file = fs.join(dataset_folder1, im_Mock)
# origData = np.array(fs.imread(dataset_file))
# origData = fs.normalize_8_bit(origData)
# label_file = fs.join(dataset_folder1, mask_Mock)
# data_info = dict(dataset_file=dataset_file, 
#             label_file=label_file)
# labelInnerFibre = np.array(fs.imread(label_file))
# if labelInnerFibre.ndim > 2:
#     labelInnerFibre = labelInnerFibre[:,:,0]
# image_size          = (64, 64)
# train_stride_step   = 8
# itera_enlarge_size  = 2
# itera_shrink_size  = 1
# fig_path            = fs.join(dataset_folder, 'crop_data')
# save_fig            = False
# sample_info         = dict(image_size = image_size, 
#                         train_stride_step=train_stride_step) 
# divided_img = divide_img_blocks(origData, n_blocks=(2,2))
# divided_label = divide_img_blocks(labelInnerFibre, n_blocks=(2,2))

# train_data = divided_img[1][0]
# label_data = divided_label[1][0]

# Data = fs.generate_training_samples(data=[train_data, label_data], 
#                             stride_step=train_stride_step, 
#                             data_shape=image_size, 
#                             itera_enlarge_size=itera_enlarge_size,
#                             fig_path=fig_path,
#                             save_fig=save_fig,
#                             show_img=True)
# Data               = Data
# image_size         = (64, 64)
# val_percent        = 0.20 # The probability to split the data as training and testing
# epochs             = 100
# batch_size         = 16
# learning_rate      = 0.001
# net_var            = 'UnetID'
# # data_aug           = {'brightness':0.3, 'contrast':0.3,
# #                       'GaussianBlur_kernel':3, 'GaussianBlur_sigma': (0.7, 1.1)}
# data_aug           = None
# preprocess_info    = dict(data_info=data_info,
#                         sample_info=sample_info)
# fs.apis.train_net(Data=Data, image_size=image_size, output_dir=output_dir, 
#             val_percent=val_percent, epochs=epochs, batch_size=batch_size,
#             learning_rate=learning_rate, net_var=net_var, 
#             data_aug=data_aug,
#             preprocess_info=preprocess_info)
