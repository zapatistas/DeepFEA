import torch
import numpy as np
from enum import Enum
from tqdm import tqdm
import gc

np.random.seed(10)
torch.manual_seed(10)
torch.set_printoptions(precision=20)

def train_epoch(train_loader,val_loader,model,loss_fn,optimizer,device,prob,loss_scale):
    model.train()
    model.to(device)

    training_loss = 0.0
    n = len(train_loader)
        
    t_loss_dis = 0
    t_loss_eff = 0 
    v_loss_dis = 0
    v_loss_eff = 0 
    print("Training...")
    with torch.autograd.set_detect_anomaly(True):
        for i, (x,y) in tqdm(enumerate(train_loader),ncols="10"):
            
            optimizer.zero_grad()
            dis_pred,eff_pred = model(x.to(device),device,prob=prob)
        
       
    
            loss_dis = loss_fn(dis_pred,y['dis_out'].to(device))
            loss_eff = loss_fn(eff_pred,y['effective_out'].to(device))
           
            loss = (loss_scale*loss_dis+ loss_scale*loss_eff)
            loss.backward()
           
            optimizer.step()
            
            y['dis_out'].cpu()
            y['effective_out'].cpu()
            t_loss_dis += loss_dis.item()
            t_loss_eff += loss_eff.item()
            training_loss += loss.item()
            
    training_loss /= (n*loss_scale)
    t_loss_eff /= n
    t_loss_dis /=n 
    val_loss = 0.0
    
    
    gc.collect()
    
    n = len(val_loader)
    print("Validating...")
    
    with torch.no_grad():
        model.eval()
        for i, (x,y) in tqdm(enumerate(val_loader)):

            dis_pred,eff_pred = model(x.to(device),device,prob=0)
            loss_dis = loss_fn(dis_pred,y['dis_out'].to(device))
            loss_eff = loss_fn(eff_pred,y['effective_out'].to(device))
            loss =  loss_scale*(loss_dis+ loss_eff)
            val_loss += loss.item()
            v_loss_dis += loss_dis.item()
            v_loss_eff += loss_eff.item()
            y['effective_out'].cpu()
            y['dis_out'].cpu()
    
    val_loss /= (n*loss_scale)
    v_loss_eff /= n
    v_loss_dis /=n 
    
    return training_loss,val_loss, t_loss_dis,t_loss_eff,v_loss_dis,v_loss_eff


def test_3d(model,x,device,is_test=False,prob=0):
    model.eval()
    model.to(device)
    dis, eff = model(x,device,is_test,prob)#.squeeze(dim=0) #pred_hat
    dis = dis.squeeze(dim=0)
    eff = eff.squeeze(dim=0)
    return dis, eff

def test_2d(model,x,device,is_test=False,prob=0):
    model.eval()
    model.to(device)
    dis, eff = model(x,device,is_test,prob)#.squeeze(dim=0) #pred_hat
    dis = dis.squeeze(dim=0)
    eff = eff.squeeze(dim=0)
    return dis, eff


class object_type(Enum):
    two_d = 0
    three_d = 1

def denormalize_single(norm,min,max,dot_a,norm_a,is_coords=False,is_3d=False):
    norm = norm.cpu()
    if len(norm.shape) == 5 and norm.shape[1] == 2 and not is_3d:
       norm = torch.flatten(norm,0,1)
    num_features = norm.shape[1]
    norm = norm.cpu()
    if norm.dim() == 5 and not is_3d:
        norm = norm.squeeze(1)
    for i in range(num_features):
        norm[:,i,:,:] = (((norm[:,i,:,:]-norm_a)*(max[i]-min[i]))/dot_a) + min[i] 
        
    if is_coords:
        norm[:,:,8,:] = torch.round(norm[:,:,8,:],decimals=2)
   
    return norm

def normalize(raw_data,min_max,dot_a,norm_a):
        features = raw_data.shape[1]
        min_axis = min_max[0]
        max_axis = min_max[1]
        
        min_boundary = raw_data[:,features-1,:].min()
        max_boundary = raw_data[:,features-1,:].max()
        for i in range(features):
            if i == min_axis.shape[0]:
                raw_data[:,i,:] = ( (raw_data[:,i,:,:] - min_boundary)*dot_a/ (max_boundary - min_boundary)) + norm_a
            else:
                raw_data[:,i,:] = ( (raw_data[:,i,:,:] - min_axis[i])*dot_a/ (max_axis[i] - min_axis[i])) + norm_a
        
        return raw_data