import os

proj_root_path = os.path.abspath(os.path.dirname(__file__))
print('project_root_path: ', proj_root_path)

img_dir = os.path.join(proj_root_path, 'logo_train_data')
# img_dir = os.path.join(proj_root_path, '../data/ISIC_2019_inputs')
label_fpath = os.path.join(proj_root_path, 'label_data/ISIC_2019_Training_GroundTruth.csv')

'''
cls_dict = {
    'bg': 0,
    'seat_belt': 1
}
'''

# image,MEL,NV,BCC,AK,BKL,DF,VASC,SCC,UNK ---> image,1,2,3,4,5,6,7,8,0
cls_dict = {
    'UNK': 0,
    'MEL': 1,
    'NV': 2,
    'BCC': 3,
    'AK': 4,
    'BKL': 5,
    'DF': 6,
    'VASC': 7,
    'SCC': 8,
}


img_h = 512
img_w = 512
img_ch = 3

MEAN_BGR = (131.7, 134.0, 168.0)
STD = 45.3


batch_size = 1
n_train_samples = 25000

train_steps = 20000
save_n_iters = 10
eval_n_iters = 10

learning_rate = 1e-3
l2_loss_lambda = 1e-5

model_save_dir = os.path.join(proj_root_path, 'run_output')
summary_save_dir = os.path.join(proj_root_path, 'train_summary')
