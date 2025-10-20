import argparse
from copyreg import pickle
import os
import torch
from time import time
import numpy as np
from models import  NEP_Net_2D 
from utils import train_epoch
from dataloaders import Effective_2D_Displacement
from torch.utils.data import DataLoader,random_split,Subset
import distutils.util

def training_loop(model,train_loss_list,val_loss_list,numEpoch,train_loader,val_loader,model_name,exp_path,
                     cur_kappa= 40, cur_gamma=0.7, scale = 1e4, lr=1e-03 ,optim_step_list = [600,700,800,1000,1200]):
    loss_fn =torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, optim_step_list,0.7)    
    init_val = 0
    prob = 1
    loss_scale = scale 
    scheduling_counter = 0
    kappa = cur_kappa
    gamma = cur_gamma
    cuda = torch.device("cuda")
    cpu = torch.device("cpu")
    device = cuda if torch.cuda.is_available() else cpu
    start_training = time()
    time_epoch = 0.0
    os.makedirs(exp_path, exist_ok=True)
    for i in range(numEpoch):
        train_loss = 0.0
        val_loss = 0.0
        
        start = time()
        
        train_loss,val_loss, t_loss_dis,t_loss_eff,v_loss_dis,v_loss_eff = train_epoch(train_loader,val_loader,model,
        loss_fn,optimizer, device,prob, loss_scale)
            
        end = time()
        model.to(cpu)
        if scheduler.get_last_lr()[0] > 1e-8:
            scheduler.step()

        if i%kappa == 0 and i > 1 and prob > 0:
            scheduling_counter += 1 
            prob = gamma**scheduling_counter
        
        if prob < 0.01:
            prob = 0
        
        if prob == 0 and train_loss < 2e-6:
            loss_scale = 10e04 
            print(f"prob 0 at {i} epoch") 

        
        if i == 0:
            init_val = val_loss
            torch.save(model.state_dict(), f'{exp_path}/{model_name}_best')
        elif init_val > val_loss:
            torch.save(model.state_dict(), f'{exp_path}/{model_name}_best')
            init_val = val_loss 
       
        
        if i == numEpoch/2 or i == numEpoch*2/3:
            torch.save(model.state_dict(), f'{exp_path}/{model_name}_e{i}')

        val_loss_list.append(f"Total val loss: {val_loss}, Val dis : {v_loss_dis}, Val eff : {v_loss_eff}")
        train_loss_list.append(f"Total train loss: {train_loss}, Train dis : {t_loss_dis}, Train eff : {t_loss_eff}")
        time_epoch += (end-start)
        print("Epoch :{0} Train_loss:{1} Val_loss:{2} in {3} seconds with probability {4}".format(i,train_loss,val_loss,end-start,prob))
        print(f"     T_dis : {t_loss_dis} T_eff: {t_loss_eff} V_dis : {v_loss_dis} V_eff: {v_loss_eff} ")

    end_training = time()
    time_epoch /= numEpoch
    training_time = np.array([f' Total training time: {end_training-start_training}',f'Average time for one epoch {time_epoch}'])
    np.savetxt(f'{exp_path}/{model_name}_training_time.txt',training_time,fmt='%s')

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--need_train',type=lambda x:bool(distutils.util.strtobool(x)),default=True)
    p.add_argument('--epochs',type=int,default=1000)
    p.add_argument('--act_out',type=str,default='tanh',help='Activation function for output layer')
    p.add_argument('--batch_size',type=int,default=32)
    p.add_argument('--lstm_act',type =str, default='tanh',help='Activation function for ConvLSTMs')
    p.add_argument('--gamma',type=float,default=0.7)
    p.add_argument('--kappa',type=int,default=40)
    p.add_argument('--scale',type=float,default=10e4)
    p.add_argument('--lr',type=float,default=1e-03)
    p.add_argument('--channels',type=lambda s: [int(item) for item in s.split(',')],default=[64,128,256],help='Number of channels in each convltsm layer')
    p.add_argument('--dim',type=int,default=2, help='2D or 3D model')
    p.add_argument('--kernel_size',type=lambda s: [int(item) for item in s.split(',')],default=[3,3,3],help='Kernel size for each convltsm layer')
    p.add_argument('--path',type=str,default='data/2D_LEM', help='Path to dataset')
    p.add_argument('--num_features',type=int,default=5, help='Number of input features 5 for 2D, 7 for 3D')
    p.add_argument('--doutput_features',type=int,default=2,help='Number of output features for displacement field')
    p.add_argument('--model_name',type=str,default='2D_NEP_Net')
    p.add_argument('--exp_path',type=str,default='experiments/2D_NEP_Net')
    p.add_argument('--frame_size',type=int,nargs='+',default=[9,9],help='Frame size for ConvLSTM layers [height,width for 2D, height,width,depth for 3D]')

    opt = p.parse_args()

    # Example usage for 2D experiments
    train_dataset = Effective_2D_Displacement(opt.path, isval=False)
    total_size = len(train_dataset)
    num_train = int(0.8*total_size)
    num_val = total_size - num_train

    train_dt,val_dt = random_split(train_dataset,[num_train,num_val])
    subset_train = [train_dt.indices[ind] for ind in range(int(len(train_dt)))]
    subset_val = [val_dt.indices[ind] for ind in range(int(len(val_dt)))]
    train_dt = Subset(train_dataset,subset_train)
    val_dt = Subset(train_dataset,subset_val)
    train_loader = DataLoader(train_dt, batch_size=opt.batch_size,shuffle=True,drop_last=False)
    val_loader = DataLoader(val_dt, batch_size=opt.batch_size,shuffle=True,drop_last=False)
    n_layers = len(opt.channels)

    padding = 1
    config = (opt.num_features,opt.doutput_features,n_layers,opt.channels,opt.kernel_size,padding,
    opt.frame_size, opt.act_out, opt.lstm_act)
    model = NEP_Net_2D(config)
    val_loss_list = [] 
    train_loss_list = []
    
    
    nepochs = opt.epochs[0] if isinstance(opt.epochs,list) else opt.epochs

    training_loop(model=model,train_loss_list=train_loss_list,val_loss_list=val_loss_list,
                  numEpoch=nepochs,train_loader=train_loader,val_loader=val_loader,
                    model_name=opt.model_name, exp_path=opt.exp_path,
                     cur_kappa=opt.kappa, cur_gamma=opt.gamma, scale=opt.scale, lr=opt.lr)

    torch.save(model.state_dict(), f'{opt.exp_path}/{opt.model_name}')
    np.savetxt(f'{opt.exp_path}/{opt.model_name}_tloss.txt',np.array(train_loss_list),fmt="%s")
    np.savetxt(f'{opt.exp_path}/{opt.model_name}_vloss.txt',np.array(val_loss_list),fmt="%s")