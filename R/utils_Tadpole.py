import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn

def convert_num(data):
    clean_data = data.astype('float32')
    clean_data = clean_data.replace(np.nan, data.median())
    return clean_data

class TadpoleSet(Dataset):
    def __init__(self, csv_path, transforms=[]):
        data = pd.read_csv(csv_path, sep=',', error_bad_lines=False, 
                           index_col=False, dtype='unicode', na_values=['', ' '],
                           usecols=['ADAS11','CDRSB', 'ABETA_UPENNBIOMK9_04_19_17', 
                                    'ADAS13','MMSE', 'RAVLT_immediate', 'Hippocampus', 'WholeBrain', 
                                    'Entorhinal', 'MidTemp', 'DX', 'AGE','FDG', 'AV45', 'TAU_UPENNBIOMK9_04_19_17',
                                    'PTAU_UPENNBIOMK9_04_19_17','ST44SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 
                                   'ST44CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'ST103CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 
                                   'ST103SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16', 'APOE4'])
        data = data[~data.DX.isnull()]
        data['APOE4'] = data['APOE4'].replace('2', '1')
        data['APOE4'] = convert_num(data['APOE4'])
        data['ADAS11'] = convert_num(data['ADAS11'])
        data['AV45'] = convert_num(data['AV45'])
        data['FDG'] = convert_num(data['FDG'])
        data['AGE'] = convert_num(data['AGE'])
        data['Entorhinal'] = convert_num(data['Entorhinal'])
        data['WholeBrain'] = convert_num(data['WholeBrain'])
        data['Hippocampus'] = convert_num(data['Hippocampus'])
        data['RAVLT_immediate'] = convert_num(data['RAVLT_immediate'])
        data['MMSE'] = convert_num(data['MMSE'])
        data['CDRSB'] = convert_num(data['CDRSB'])
        data['ADAS13'] = convert_num(data['ADAS13'])
        data['MidTemp'] = convert_num(data['MidTemp'])
        data['ST44SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16'] = convert_num(data['ST44SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16'])
        data['ST44CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16'] = convert_num(data['ST44CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16'])
        data['ST103CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16'] = convert_num(data['ST103CV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16'])
        data['ST103SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16'] = convert_num(data['ST103SA_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16'])
        data['TAU_UPENNBIOMK9_04_19_17'] = data['TAU_UPENNBIOMK9_04_19_17'].replace('>1300', '1300')
        data['TAU_UPENNBIOMK9_04_19_17'] = data['TAU_UPENNBIOMK9_04_19_17'].replace('<80', '80')
        data['TAU_UPENNBIOMK9_04_19_17'] = convert_num(data['TAU_UPENNBIOMK9_04_19_17'])
        data['PTAU_UPENNBIOMK9_04_19_17'] = data['PTAU_UPENNBIOMK9_04_19_17'].replace('>120', '120')
        data['PTAU_UPENNBIOMK9_04_19_17'] = data['PTAU_UPENNBIOMK9_04_19_17'].replace('<8', '8')
        data['PTAU_UPENNBIOMK9_04_19_17'] = convert_num(data['PTAU_UPENNBIOMK9_04_19_17'])
        data['ABETA_UPENNBIOMK9_04_19_17'] = data['ABETA_UPENNBIOMK9_04_19_17'].replace('<200', '200')
        data['ABETA_UPENNBIOMK9_04_19_17'] = convert_num(data['ABETA_UPENNBIOMK9_04_19_17'])
        
        DX = data['DX']
        del data['DX']
        DX = DX.replace('NL to MCI', 'MCI')
        DX = DX.replace('MCI to Dementia', 'Dementia')
        DX = DX.replace('Dementia to MCI', 'MCI')
        DX = DX.replace('NL to Dementia', 'Dementia')
        DX = DX.replace('MCI to NL', 'NL')
        
        self.labels, self.labnames = pd.factorize(DX)
        self.labels = self.labels.astype('int64')
        self.prednames = list(data.columns)
        self.predictors = data.values
        
    def __getitem__(self, index):
        label = self.labels[index]
        preds = self.predictors[index, :]
        return preds, label
    
    def __len__(self):
        return len(self.labels)
    
class TadpoleModule(nn.Module):
    def __init__(self, transforms=[]):
        super().__init__()
        self.transforms = transforms
        self.l1 = nn.Linear(20, 10)
        self.l2 = nn.Linear(10, 3)
        self.r1 = nn.ReLU()
        
    def forward(self, x):
        ret = x
        for t in self.transforms:
            ret = t(ret)
        ret = self.l1(ret)
        ret = self.r1(ret)
        ret = self.l2(ret)
        return ret
    
class TadpoleSimple(nn.Module):
    def __init__(self, transforms=[]):
        super().__init__()
        self.transforms = transforms
        self.layer = nn.Linear(20, 3)
        
    def forward(self, x):
        ret = x
        for t in self.transforms:
            ret = t(ret)
        ret = self.layer(ret)
        return ret

class Normalizer():
    ''' Clase para normalizar por loader y batch '''
    def __init__(self, loader):
        n = 0
        mean = 0
        mean2 = 0
        for data, _ in loader:
            mean += data.mean(dim=0)
            mean2 += (data**2).mean(dim=0)  #dim=0 suma por filas, dim=1 columnas
        self.mean = mean
        self.std = torch.sqrt(mean/n - mean**2)
    
    def transform(self, x):
        return (x - self.mean)/self.std
    
    def untransform(self, x):
        return (x * self.std) + self.mean
    
    def __call__(self, x):
        return self.transform(x)
    
def train(epoch, model, loader, criterion, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        #data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({}%)]\tLoss: {}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                100. * batch_idx / len(loader), loss.item()))
    return loss.item()

def val(model, loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    corrects = []
    for data, target in loader:
        #data, target = Variable(data), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).item()
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        
    test_loss /= len(loader.dataset)
    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({}%)\n'.format(
        test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))
    return test_loss, correct.item()