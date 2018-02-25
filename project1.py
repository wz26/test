
# coding: utf-8

# In[28]:

import numpy as np


# In[29]:

#B matrix is an 12-by-12 matrix to find parameter value
B = np.zeros((12,12), dtype=int)


# In[30]:

#this  six points are used to obtain the intrinsics and extrinsics
xw1, yw1, zw1, x1, y1 = 6, 0, 2, 793, 884
xw2, yw2, zw2, x2, y2 = 0, 6, 2, 1080, 880
xw3, yw3, zw3, x3, y3 = 4, 0, 2, 845, 884
xw4, yw4, zw4, x4, y4 = 0, 4, 2, 1033, 880
xw5, yw5, zw5, x5, y5 = 6, 0, 0, 793, 950
xw6, yw6, zw6, x6, y6 = 0, 6, 0, 1080, 948


# In[31]:

#create matrix B
B[0] = [xw1, yw1, zw1, 1, 0, 0, 0, 0, -x1*xw1, -x1*yw1, -x1*zw1, -x1]
B[1] = [0, 0, 0, 0, xw1, yw1, zw1, 1, -y1*xw1, -y1*yw1, -y1*zw1, -y1]
B[2] = [xw2, yw2, zw2, 1, 0, 0, 0, 0, -x2*xw2, -x2*yw2, -x2*zw2, -x2]
B[3] = [0, 0, 0, 0, xw2, yw2, zw2, 1, -y2*xw2, -y2*yw2, -y1*zw2, -y2]
B[4] = [xw3, yw3, zw3, 1, 0, 0, 0, 0, -x3*xw3, -x3*yw3, -x3*zw3, -x3]
B[5] = [0, 0, 0, 0, xw3, yw3, zw3, 1, -y3*xw3, -y3*yw3, -y3*zw3, -y3]
B[6] = [xw4, yw4, zw4, 1, 0, 0, 0, 0, -x4*xw4, -x4*yw4, -x4*zw4, -x4]
B[7] = [0, 0, 0, 0, xw4, yw4, zw4, 1, -y4*xw4, -y4*yw4, -y4*zw4, -y4]
B[8] = [xw5, yw5, zw5, 1, 0, 0, 0, 0, -x5*xw5, -x5*yw5, -x5*zw5, -x5]
B[9] = [0, 0, 0, 0, xw5, yw5, zw5, 1, -y5*xw5, -y5*yw5, -y5*zw5, -y5]
B[10] = [xw6, yw6, zw6, 1, 0, 0, 0, 0, -x6*xw6, -x6*yw6, -x6*zw6, -x6]
B[11] = [0, 0, 0, 0, xw6, yw6, zw6, 1, -y6*xw6, -y6*yw6, -y6*zw6, -y6]


# In[32]:

#do svd and get the last column of vh.T to obtain projection matrix
u, s, vh = np.linalg.svd(B, full_matrices=True)
vh = vh.T
p = vh[:,11]
p = p.reshape(3, 4)


# In[33]:

#extract left and right part of projection matrix
right = p[:, 3]
left = p[:, [0,1,2]]


# In[34]:

#normalizing the projection matrix
temp = left[2, :]
temp
norm = np.sum(temp[0]**2 + temp[1]**2 + temp[2]**2)
norm = np.sqrt(norm)
p = p/norm
right = p[:, 3]
left = p[:, [0,1,2]]


# In[35]:

#compute the matrix A to recover intrinsics
A = np.matmul(left, left.transpose())
A = A/A[2,2]


# In[36]:

# recovering the intrinsics
u0 = A[0,2]
v0 = A[1,2]
beta = np.sqrt(A[1,1]-np.square(v0))
s=0
alfa = np.sqrt(A[0,0]-np.square(u0)-np.square(s))


# In[37]:

#recovering the extrinsics
K = np.matrix([[alfa, s, u0],[0, beta, v0],[0, 0, 1]])
Kinverse = K.I
R = np.matmul(Kinverse, left)
t = np.matmul(Kinverse, right)
RT = np.concatenate((R, t.T), axis=1)


# In[38]:

#now begin to verify, firsly pick a points previously used to verify
#xw1, yw1, zw1, x1, y1 = 6, 0, 2, 793, 884
X1 = np.array([10, 0, 6, 1]);
x1 = np.matmul(K, RT)
x1 = np.matmul(x1, X1.T)
x1 = x1/x1[0,2]

#create an array of 8 points to verify
verify = np.zeros((8,5), dtype=int)
#first two columns are for image coordinates, last three columns for 3-D coordinates
verify[0] = [1031, 744, 0 , 4 , 6 ]
verify[1] = [1033, 811, 0 , 4 , 4 ]
verify[2] = [1342, 437, 0 , 16, 14]
verify[3] = [1128, 465, 0 , 8 , 14]
verify[4] = [518 , 445, 16, 0 , 14]
verify[5] = [740 , 471, 8 , 0 , 14]
verify[6] = [795 , 748, 6 , 0 , 6 ]
verify[7] = [688 , 742, 10, 0 , 6 ]


# In[43]:

#use eight points to verify 
dist = 0
result = np.zeros((8,2), dtype=float)
for i in range(0, 8):
    X = np.array([verify[i,2], verify[i,3], verify[i,4], 1])
    x = np.matmul(K, RT)
    x = np.matmul(x, X.T)
    x = x/x[0, 2]
    temp = 0
    temp += np.square(x[0, 0] - verify[i,0])
    temp += np.square(x[0, 1] - verify[i,1])
    dist += np.sqrt(temp)

result
dist /= 8      
dist

print("The average error distance is: %.2f"%(dist))
print("The extrisinc matrix K is: ")
print(np.matrix(K))
print("The rotation matrix R is: ")
print(R)
print("The transportation matrix t is: ")
print(t)


# In[ ]:




# In[ ]:



