import matplotlib.pyplot as plt
import numpy as np

single=np.load('single_64_64.npy')
multi = np.load('basic_64_64.npy')
#space_var,time_var,psnr_list,bilinear_psnr_list

space_var=single[0,:]
time_var=single[1,:]

b_psnr = single[3,:]
s_psnr = single[2,:]
m_psnr = multi[2,:]
print(np.mean(b_psnr),np.mean(s_psnr),np.mean(m_psnr))

space_psnr = s_psnr-b_psnr
box_idx_1=np.zeros_like(space_var,dtype=int)
box_idx_1[:]=-1
for i in range(len(space_var)):
    idx = space_var[i]//250
    if idx>=10:
        continue
    box_idx_1[i]=int(idx)
box_data_1 = [[] for i in range(10)]
for i in range(len(space_var)):
    if box_idx_1[i]<0:
        continue
    
    box_data_1[box_idx_1[i]].append(space_psnr[i])

plt.boxplot(box_data_1,showfliers=False,notch =True)
plt.savefig("box")
plt.close()
print(m_psnr)
# plt.scatter(space_var,b_psnr,c='red',s=0.2)

plt.scatter(space_var,m_psnr-s_psnr,c='green',s=0.4)
# plt.scatter(space_var,m_psnr-b_psnr,c='blue',s=0.4)
plt.show()
plt.savefig("s_psnr.png")
plt.close()

time_psnr = s_psnr
box_idx_2=np.zeros_like(time_var,dtype=int)
box_idx_2[:]=-1
for i in range(len(time_var)):
    idx = time_var[i]//500
    if idx>=10:
        continue
    box_idx_2[i]=int(idx)
box_data_2 = [[] for i in range(10)]
for i in range(len(time_var)):
    if box_idx_2[i]<0:
        continue
    
    box_data_2[box_idx_2[i]].append(time_psnr[i])


plt.boxplot(box_data_2,showfliers=False)
plt.savefig("box2")
plt.close()

plt.scatter(time_var,m_psnr-s_psnr,c='green',s=0.05)
# plt.scatter(time_var,m_psnr-b_psnr,c='blue',s=0.4)
plt.show()
plt.savefig("t_psnr.png")
plt.close()

ax = plt.axes(projection ="3d")
ax.scatter3D(time_var, space_var, m_psnr-s_psnr, color = "red", s=0.1)
plt.title("3D scatter plot")
plt.show()
plt.savefig("3d.jpg")
