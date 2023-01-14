import numpy as np
import os
import re

device={0:"cpu",1:"gpu",2:"nnapi",3:"dsp"}


def get_speed_data(file, device_name):

    feature_list=[[],[],[],[]]
    feature_zero_list=[{},{},{},{}]
    
    f=open(file, 'r')
    lines = f.readlines()
    for line in lines:
        line_ = line.strip("\n")
        line_ = line_.split(" ")
        # print(line_)
        dirname=line_[0]
        
        filename=line_[1]
        filename=filename.split("_")
        if filename[1]=='zero':
            if dirname.split("/")[1]!='normal' or dirname.split("/")[2]!='seperate':
                continue
            feature_i=[]
            channel = int(filename[2].split(".")[0])
            # feature_i.append(int(filename[2].split(".")[0]))
            # feature_i.append(0.0)
            # feature_i.append(0.0)
            processor = int(line_[2][-1])
            
            time = (line_[3].split(":")[-1])
            time = float(time)/1000000
            if processor in [0,1,2,3]:
                feature_zero_list[processor][channel]=time
            else:
                print("no device found")
                exit()
        else:
            feature_i=[]
            seperate =int(filename[6][0])
            
            
            if seperate==0 or filename[1]=='inverted':
                continue
            
            
            # if filename[1]=='normal':
            #     feature_i.append(0)
            # else:
            #     feature_i.append(1)
            
            feature_i.append(int(filename[2]))
            feature_i.append(int(filename[2])-int(filename[4]))
            feature_i.append(int(filename[5]))

            processor = int(line_[2][-1])
            
            time = (line_[3].split(":")[-1])
            feature_i.append(float(time)/1000000)
            if processor in [0,1,2,3]:
                feature_list[processor].append(feature_i)
            else:
                print("no device found")
                exit()
    if os.path.exists(device_name)==0:
        os.mkdir(device_name)
    feature_sum = np.zeros((0,4))
    for i in range(4):
        if len(feature_list[i])>0:
            feature = np.array(feature_list[i])
            for j in range(feature.shape[0]):
                feature[j,3] -= feature_zero_list[i][feature[j,0]]
                feature[j,3] /= 4
            
            np.save(os.path.join(device_name,device[i]),feature)
            print(feature.shape)
    
    print(feature_sum.shape)

        
#Input channel, Skip channel, kernel size, processor, time#
if __name__ == "__main__":

    get_speed_data('myFile_test.txt',"huawei_p30")