import os
from torch.utils.data import TensorDataset
from utils import normalize
import torch
import numpy as np
np.random.seed(10)
torch.manual_seed(10)
torch.set_printoptions(precision=20)

class Effective_3D_Displacement(TensorDataset):   
    def __init__(self,dataset_path, isval=False):
        # Load min max values for normalization
        # Need to be in the dataset folder
        min_f =  np.load(f"{dataset_path}/dis_f_min.npy")
        max_f =  np.load(f"{dataset_path}/dis_f_max.npy")
        min_c =  np.load(f"{dataset_path}/dis_c_min.npy")
        max_c =  np.load(f"{dataset_path}/dis_c_max.npy")
        min_strain = np.load(f"{dataset_path}/eff_strain_min.npy")
        max_strain = np.load(f"{dataset_path}/eff_strain_max.npy").item()
        min_stress = np.load(f"{dataset_path}/eff_stress_min.npy")
        max_stress = np.load(f"{dataset_path}/eff_stress_max.npy").item()
        min_strain = 0 
        min_stress = 0
        self.min_max_in = [np.concatenate((min_c,min_f)), np.concatenate((max_c,max_f))]
        self.min_max_out = [np.concatenate((np.expand_dims(min_c,0),np.expand_dims(max_c,0)),axis=0), [np.array([min_stress,min_strain]), np.array([max_stress,max_strain])]]
        self.path = dataset_path
        self.num_steps = 1     
        norm_max = 1
        norm_min = -1
        self.isval = isval
        self.dot_a = norm_max - norm_min
        self.norm_a = norm_min
        self.is_test = False 
        if os.path.exists(dataset_path):
            self.len = len(next(os.walk(dataset_path))[1])
        else:
            print("No dataset folder found")
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        
        x = np.load(f"{self.path}/{index+1}/dis_in.npz")['arr_0']
        
        y = {'dis_out': None, 'effective_out' : None}

        y['dis_out'] = np.load(f"{self.path}/{index+1}/dis_out.npz",allow_pickle=True)['arr_0']
        try:
            eff = np.load(f"{self.path}/{index+1}/effective_out.npz",allow_pickle=True)
        except:
            print(f"Error loading effective stress and strain for {index+1} mesh")
        eff_stress = eff['eff_stress']
        eff_strain = eff['eff_strain']
       
        y['effective_out'] = np.concatenate((eff_stress,eff_strain), axis = 1 )
        x = torch.tensor(normalize(x,self.min_max_in,self.dot_a,self.norm_a).astype(np.single))
        y['dis_out'] = torch.tensor(normalize(y['dis_out'],self.min_max_out[0],self.dot_a,self.norm_a).astype(np.single))
        y['effective_out'] = torch.tensor(normalize(y['effective_out'],self.min_max_out[1],self.dot_a,self.norm_a).astype(np.single))
       
        return x,y

class Effective_2D_Displacement(TensorDataset):   
    def __init__(self,dataset_path, isval=False):
        # Load min max values for normalization
        # Need to be in the dataset folder
        min_f =  np.load(f"{dataset_path}/dis_f_min.npy")
        max_f =  np.load(f"{dataset_path}/dis_f_max.npy")
        min_c =  np.load(f"{dataset_path}/dis_c_min.npy")
        max_c =  np.load(f"{dataset_path}/dis_c_max.npy")
        min_strain = np.load(f"{dataset_path}/eff_strain_min.npy")
        max_strain = np.load(f"{dataset_path}/eff_strain_max.npy").item()
        min_stress = np.load(f"{dataset_path}/eff_stress_min.npy")
        max_stress = np.load(f"{dataset_path}/eff_stress_max.npy").item()
        min_strain = 0 
        min_stress = 0
        self.min_max_in = [np.concatenate((min_c,min_f)), np.concatenate((max_c,max_f))]
        self.min_max_out = [np.concatenate((np.expand_dims(min_c,0),np.expand_dims(max_c,0)),axis=0), [np.array([min_stress,min_strain]), np.array([max_stress,max_strain])]]
        self.path = dataset_path
        self.num_steps = 1     
        norm_max = 1
        norm_min = -1
        self.isval = isval
        self.dot_a = norm_max - norm_min
        self.norm_a = norm_min
        if os.path.exists(dataset_path):
            self.len = len(next(os.walk(dataset_path))[1])
        else:
            print("No dataset folder found")
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
          
        x = np.load(f"{self.path}/{index+1}/dis_in.npz",allow_pickle=True)['arr_0']
        y = {'dis_out': None, 'effective_out' : None}
        y['dis_out'] = np.load(f"{self.path}/{index+1}/dis_out.npz",allow_pickle=True)['arr_0']
        try:
            eff = np.load(f"{self.path}/{index+1}/effective_out.npz",allow_pickle=True)
        except:
            print(f"Error loading effective stress and strain for {index+1} mesh")
        
        eff_stress = eff['eff_stress']
        eff_strain = eff['eff_strain']
       
       
        y['effective_out'] = np.concatenate((eff_stress,eff_strain), axis = 1 )
        x = torch.tensor(normalize(x,self.min_max_in,self.dot_a,self.norm_a).astype(np.single))
        y['dis_out'] = torch.tensor(normalize(y['dis_out'],self.min_max_out[0],self.dot_a,self.norm_a).astype(np.single))
        y['effective_out'] = torch.tensor(normalize(y['effective_out'],self.min_max_out[1],self.dot_a,self.norm_a).astype(np.single))
        return x,y


