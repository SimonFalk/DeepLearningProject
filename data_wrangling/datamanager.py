import numpy as np
import scipy.io as sio
import os
from h5py import File as h5pyFile

class DataLoader:
    
    def __init__(self, root = '/media/big/BigDownloads/'):
        self.root = root
        if not root.endswith('/'):
            self.root += '/'
            
    def get_data_descriptions(self):
        return {
            'bf1': '''
            
                bf = Basal Forebrain.
            
                LFP recordings from rats in two conditions.
                (a) Resting condition in home cage.
                (b) Exploration of novel arena.
                Waking state only - no sleep epochs.
                Sampling rate 400 Hz.
                Times somewhat irregular; very short.
                Includes some data relating to movement (extra).
                
                Basal forebrain contributes to default mode network regulation
                Jayakrishnen Nair, Arndt-Lukas Klaassen, Jozsef Arato, Alexei L. Vyssotski,
                Michael Harvey, Gregor Rainer
                PNAS (2018) doi:10.1073/pnas.1712431115
                
                Jayakrishnen Nair, Arndt-Lukas Klaassen, Jozsef Arato, Alexei L. Vyssotski, Michael
                Harvey, Gregor Rainer (2018); Basal forebrain LFP recordings from rats in home cage
                and during arena exploration. CRCNS.org.
                http://dx.doi.org/10.6080/K0MK6B2Q
                
                http://crcns.org/data-sets/bf/bf-1/about-bf-1
            ''',
            
            'fcx2': '''
                
                fcx = Frontal Cortex.
                
                Intracranial EEG recordings; 10 human adults.
                Subjects are performing visuospatial working memory task.
                Subjects are epileptic.
                Sampling rate 512 Hz or 1 kHz.
                
                Johnson, E. L., Adams, J. N., Solbakk, A.-K., Endestad, T., Larsson, P. G.,
                Ivanovic, J., Meling, T. R., Lin, J. J., Knight, R. T. Dynamic frontotemporal
                systems process space and time in working memory. PLOS Biology 16,
                e2004274 (2018). doi:10.1371/journal.pbio.2004274
                
                Johnson (2018); Intracranial EEG recordings of medial temporal, lateral frontal, and
                orbitofrontal regions in 10 human adults performing a visuospatial working memory task.
                CRCNS.org
                http://dx.doi.org/10.6080/K0VX0DQD
                
                In case of publication; additional: eljohnson@berkeley.edu
                
                Extras are messy and NOT implemented.
                
                http://crcns.org/data-sets/fcx/fcx-2/about-fcx-2
                
            '''
        }
        
    def get_bf1(
        self, 
        sessions = 'all', 
        extra = False,
        path = 'bf-1/basal_forebrain_lfp/'
    ):
        
        if sessions == 'all':
            sessions = os.listdir(f'{self.root}{path}')
            
        datas = [sio.loadmat(f'{self.root}{path}{f}') for f in sessions]
        
        if not extra:
            return {
                f.replace('.mat', '').split('/')[-1]: {
                    'data': obj['LFP_data'],
                    'time': obj['T_data'],
                    'extra': obj['mov_data']
                }
                for obj, f in zip(datas, sessions)
            }
        
        return {
            f.replace('.mat', '').split('/')[-1]: {
                'data': obj['LFP_data'],
                'time': obj['T_data']
            }
            for obj, f in zip(datas, sessions)
        }
    
    
    def get_fcx2(
        self,
        sessions = 'all',
        path = 'fcx-2/'
    ):
        
        if sessions == 'all':
            sessions = [f's{no}' for no in range(1, 11)]
            
        sessions = [f'{self.root}{path}{s}' for s in sessions]
        tmp = []
        for s in sessions:
            for p in os.listdir(s):
                if 'data_primary' in p:
                    tmp.append(f'{s}/{p}')
        sessions = tmp
        del tmp
            
        datas = {}
        for s in sessions:
            with h5pyFile(s, 'r') as fp:
                data = np.array(fp['gdat_clean_filt'])
                # Errors 1: empty channels
                data = data[:, (data == 0).all(0) == False]
                # Errors 2: Sessions that are padded with zeroes to extreme lengths
                #   (Note: == 0 comparisson works, since no non-trash-value is close enough to 0 to return true)
                data = data[data != 0].reshape((-1, data.shape[1]))
                
                datas[s.split('/')[-2]] = {
                    'data': data
                }
                
        return datas