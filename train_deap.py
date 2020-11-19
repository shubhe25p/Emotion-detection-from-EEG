import _pickle as cPickle
import os
from multiprocessing import Pool
from pathlib import Path
import sys
import numpy as np
import time
import itertools

chan = ['Fp1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1','P3','P7','PO3','O1','Oz','Pz','Fp2','AF4','Fz','F4','F8','FC6','FC2','Cz','C4','T8','CP6','CP2','P4','P8','PO4','O2']
nLabel, nTrial, nUser, nChannel, nTime  = 4, 40, 32, 32, 8064
print("Program started \n")

class trainEmotion(object): 

    def do_fft(self,all_channel_data): 
    	"""
    	Do fft in each channel for all channels.
    	Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
    	Output: FFT result with dimension N x M. N denotes number of channel and M denotes number of FFT data from each channel.
    	"""
    	data_fft = map(lambda x: np.fft.fft(x),all_channel_data)
    	return data_fft
    def get_frequency(self,all_channel_data): 
    	"""
    	Get frequency from computed fft for all channels. 
    	Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
    	Output: Frequency band from each channel: Delta, Theta, Alpha, Beta, and Gamma.
    	"""
    	#Length data channel
    	L = len(all_channel_data[0])
    	#Sampling frequency
    	Fs = 128
    	#Get fft data
    	data_fft = self.do_fft(all_channel_data)
    	#Compute frequencymotio
    	frequency = map(lambda x: abs(x//L),data_fft)
    	frequency = map(lambda x: x[: L//2+1]*2,frequency)
    	f1,f2,f3,f4,f5=itertools.tee(frequency,5)
    	#List frequency
    	delta = np.array(list(map(lambda x: x[L*1//Fs-1: L*4//Fs],f1)))
    	theta = np.array(list(map(lambda x: x[L*4//Fs-1: L*8//Fs],f2)))
    	alpha = np.array(list(map(lambda x: x[L*5//Fs-1: L*13//Fs],f3)))
    	beta =  np.array(list(map(lambda x: x[L*13//Fs-1: L*30//Fs],f4)))
    	gamma = np.array(list(map(lambda x: x[L*30//Fs-1: L*50//Fs],f5)))
    	return delta,theta,alpha,beta,gamma
    
    
    def get_feature(self,all_channel_data): 
        (delta,theta,alpha,beta,gamma) = self.get_frequency(all_channel_data)
        delta_std = np.std(delta,axis=1)
        theta_std = np.std(theta,axis=1)
        alpha_std = np.std(alpha,axis=1)
        beta_std = np.std(beta,axis=1)
        gamma_std = np.std(gamma,axis=1)
        feature = np.array([delta_std,theta_std,alpha_std,beta_std,gamma_std])
        feature = feature.T
        feature = feature.ravel()
    	
        return feature
		#send emotion_class to web app
    
    def main_process(self):
        fname=Path("data/s01.dat")
        fout_data = open("features_std.csv",'w')
        va_label = open("class_valence.csv","w")
        ar_label = open("class_arousal.csv","w")
        x = cPickle.load(open(fname, 'rb'), encoding="bytes")
        for i in range(40):
            eeg_realtime=x[b'data'][i]
            label=x[b'labels'][i] 
            if(label[0]>6):
                val_v=3
            elif(label[0]<4):
                val_v=1
            else:
                val_v=2
            
            if(label[1]>6):
                val_a=3
            elif(label[1]<4):
                val_a=1
            else:
                val_a=2
             
            if(i<39):
                va_label.write(str(val_v)+",")
                ar_label.write(str(val_a)+",")
            else:
                va_label.write(str(val_v))
                ar_label.write(str(val_a))
            
            eeg_raw=np.reshape(eeg_realtime,(40,8064))
            eeg_raw=eeg_raw[:32,:]
            eeg_feature_arr=self.get_feature(eeg_raw)
            for f in range(160):
                if(f==159):
                    fout_data.write(str(eeg_feature_arr[f]))
                else:
                    fout_data.write(str(eeg_feature_arr[f])+",")
            fout_data.write("\n")
            print(str(i)+" Video watched")

if __name__ == "__main__":
	rte = trainEmotion()
	rte.main_process()
	

    
    
    