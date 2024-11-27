# -*- coding: utf-8 -*-
"""

@author: Monica Visintin

Regress Total UPDRS from the other features in file "parkinsons_updrs.csv"

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.precision', 3)
#%% Read the input csv file
plt.close('all') # close all the figures that might still be open from previous runs
x=pd.read_csv("parkinsons_updrs.csv") # read the dataset; x is a Pandas dataframe
features=list(x.columns)#  list of features in the dataset
subj=pd.unique(x['subject#'])# existing values of patient ID
print("The original dataset shape  is ",x.shape)
print("The number of distinct patients in the dataset is ",len(subj))
print("the original dataset features are ",len(features))
print(features)

#%% Group measurements performed in the same day and find the mean
X=pd.DataFrame()
for k in subj:
    xk=x[x['subject#']==k]# data of user k
    xk1=xk.copy()# we modify the values of xk (next lines); a warning would be issued if we did not make a copy
    xk1.test_time=xk1.test_time.astype(int)# remove decimal values
    xk1['g']=xk1['test_time']# add a new feature
    v=xk1.groupby('g').mean()# group according to the new feature (it is removed)
    X=pd.concat([X,v],axis=0,ignore_index=True)# append the new data to xx
features=list(x.columns)
print("The dataset shape after the mean is ",X.shape)
print("The features of the dataset are ",len(features))
#print(features)
Np,Nc=X.shape# Np = number of rows/ptients Nc=number Nf of regressors + 1 (regressand total UPDRS is included)
#%% Have a look at the dataset
print(X.describe().T) # gives the statistical description of the content of each column
print(X.info())
#%% Measure and show the covariance matrix
Xnorm=(X-X.mean())/X.std()# normalized data
c=Xnorm.cov()# note: xx.cov() gives the wrong result
plt.figure()
plt.matshow(np.abs(c.values),fignum=0)# absolute value of corr.coeffs
plt.xticks(np.arange(len(features)), features, rotation=90)
plt.yticks(np.arange(len(features)), features, rotation=0)    
plt.colorbar()
plt.title('Correlation coefficients of the features')
plt.tight_layout()
plt.savefig('./corr_coeff.png') # save the figure
plt.show()
plt.figure()
c.total_UPDRS.plot()
plt.grid()
plt.xticks(np.arange(len(features)),features,rotation=90)
plt.title('Corr. coeff. between total_UPDRS and the other features')
plt.tight_layout()
plt.show()
plt.savefig('./UPDRS_corr_coeff.png') # save the figure
#%% Shuffle the data (two out of many methods)
# first method:
np.random.seed(307774) # set the seed for random shuffling
indexsh=np.arange(Np) # generate array [0,1,...,Np-1]
np.random.shuffle(indexsh) # shuffle the array
Xsh=X.copy()
Xsh=Xsh.set_axis(indexsh,axis=0,inplace=False) # shuffle accordingly the dataframe
Xsh=Xsh.sort_index(axis=0) # reset index of the dataframe
# comment: Xsh.reset_index() exists, but a further index column would be created
# second method
#Xsh=X.sample(frac=1, replace=False, random_state=30, axis=0, ignore_index=True)
#Xsh=X.sample(frac=1, replace=False, axis=0, ignore_index=True)

#%% Generate training and test matrices
Ntr=int(Np*0.5)  # number of training points
Nte=Np-Ntr   # number of test points
#%% evaluate mean and st.dev. for the training data only
X_tr=Xsh[0:Ntr]# dataframe that contains only the training data
mm=X_tr.mean()# mean (series)
ss=X_tr.std()# standard deviation (series)
my=mm['total_UPDRS']# mean of total UPDRS
sy=ss['total_UPDRS']# st.dev of total UPDRS
#%% Generate the normalized training and test datasets, remove unwanted regressors
Xsh_norm=(Xsh-mm)/ss# normalized data
ysh_norm=Xsh_norm['total_UPDRS']# regressand only
Xsh_norm=Xsh_norm.drop(['total_UPDRS','subject#','Jitter:DDP','Shimmer:DDA'],axis=1)# regressors only
regressors=list(Xsh_norm.columns)
Nf = len(regressors) # number of regressors
print (Nf)
print("The new regressors are: ",len(regressors))
#print(regressors)
Xsh_norm=Xsh_norm.values # from dataframe to Ndarray
ysh_norm=ysh_norm.values # from dataframe to Ndarray
X_tr_norm=Xsh_norm[0:Ntr]
X_te_norm=Xsh_norm[Ntr:]
y_tr_norm=ysh_norm[0:Ntr]
y_te_norm=ysh_norm[Ntr:]
print(X_tr_norm.shape,X_te_norm.shape)
#%% LLS regression
w_hat=np.linalg.inv(X_tr_norm.T@X_tr_norm)@(X_tr_norm.T@y_tr_norm)
y_hat_tr_norm=X_tr_norm@w_hat
y_hat_te_norm=X_te_norm@w_hat
#%% de-normalize data
y_tr=y_tr_norm*sy+my
y_te=y_te_norm*sy+my
y_hat_tr=y_hat_tr_norm*sy+my
y_hat_te=y_hat_te_norm*sy+my
#%% Steepest descent regression
H = 2*X_tr_norm.T@X_tr_norm #Hessain H(w^(i))=2X.T@X
i = 0
def Steepest_descent(X_tr_norm,y_tr_norm,i):
    W = []
    W_hat_SDi = np.zeros(X_tr_norm.shape[1]) #create zero vector for making the while loop works for the first time for the w_hat(i)      
    W_hat_SDi1 = np.random.rand(X_tr_norm.shape[1]) #random value for W_hat(i+1)
    while (np.linalg.norm(W_hat_SDi1-W_hat_SDi) > 0.0000001):
        
        W += [W_hat_SDi] #i = w_hat(i)
        W_hat_SDi = W_hat_SDi1 #i1 = w_hat(i+1)
        gradient = -2*X_tr_norm.T@y_tr_norm + 2*X_tr_norm.T@X_tr_norm@W_hat_SDi
        gamma = (np.inner(gradient, gradient) / (gradient.T@H@gradient))
        W_hat_SDi1 = W_hat_SDi -( gamma * gradient)
        i +=1
        
    return(W_hat_SDi)

W_hat_SD = Steepest_descent(X_tr_norm,y_tr_norm,i)
print ("---------->>>>>>>>",W_hat_SD)
#%% plot the optimum weight vector for Steepest descent
nn=np.arange(Nf)
plt.figure(figsize=(6,4))
plt.plot(nn,W_hat_SD,'-o')
ticks=nn
plt.xticks(ticks, regressors, rotation=90)
plt.ylabel(r'$\^w(n)$')
plt.title('Steepest Descent-Optimized weights')
plt.grid()
plt.tight_layout()
plt.savefig('./Steepest Descent-what.png')
plt.show()
#%%
y_hat_tr_norm_SD=X_tr_norm@W_hat_SD
y_hat_te_norm_SD=X_te_norm@W_hat_SD

#%%
y_hat_tr_SD=y_hat_tr_norm_SD*sy+my
y_hat_te_SD=y_hat_te_norm_SD*sy+my

#%% plot the error histograms Steepest descent
E_tr_SD=(y_tr-y_hat_tr_SD)# training
E_te_SD=(y_te-y_hat_te_SD)# test
M=np.max([np.max(E_tr_SD),np.max(E_te_SD)])
m=np.min([np.min(E_tr_SD),np.min(E_te_SD)])
common_bins=np.arange(m,M,(M-m)/50)
e=[E_tr_SD,E_te_SD]
plt.figure(figsize=(6,4))
plt.hist(e,bins=common_bins,density=True, histtype='bar',label=['training','test'])
plt.xlabel(r'$e=y-\^y$')
plt.ylabel(r'$P(e$ in bin$)$')
plt.legend()
plt.grid()
plt.title('SD-Error histograms using all the training dataset')
plt.tight_layout()
plt.savefig('./SD-hist.png')
plt.show()
#%% plot the regression lines Steepest_Descent
plt.figure(figsize=(4,4))
plt.plot(y_te,y_hat_te_SD,'.',label='all')
plt.legend()
v=plt.axis()
plt.plot([v[0],v[1]],[v[0],v[1]],'r',linewidth=2)
plt.xlabel(r'$y$')
plt.axis('square')
plt.ylabel(r'$\^y$')
plt.grid()
plt.title('Steepest_Descent-test')
plt.tight_layout()
plt.savefig('./Steepest_Descent-yhat_vs_y.png')
plt.show()
#%% plot the optimum weight vector for LLS
nn=np.arange(Nf)
plt.figure(figsize=(6,4))
plt.plot(nn,w_hat,'-o')
ticks=nn
plt.xticks(ticks, regressors, rotation=90)
plt.ylabel(r'$\^w(n)$')
plt.title('LLS-Optimized weights')
plt.grid()
plt.tight_layout()
plt.savefig('./LLS-what.png')
plt.show()
#%% plot the error histograms
E_tr=(y_tr-y_hat_tr)# training
E_te=(y_te-y_hat_te)# test
M=np.max([np.max(E_tr),np.max(E_te)])
m=np.min([np.min(E_tr),np.min(E_te)])
common_bins=np.arange(m,M,(M-m)/50)
e=[E_tr,E_te]
plt.figure(figsize=(6,4))
plt.hist(e,bins=common_bins,density=True, histtype='bar',label=['training','test'])
plt.xlabel(r'$e=y-\^y$')
plt.ylabel(r'$P(e$ in bin$)$')
plt.legend()
plt.grid()
plt.title('LLS-Error histograms using all the training dataset')
plt.tight_layout()
plt.savefig('./LLS-hist.png')
plt.show()
#%% plot the regression lines
plt.figure(figsize=(4,4))
plt.plot(y_te,y_hat_te,'.',label='all')
plt.legend()
v=plt.axis()
plt.plot([v[0],v[1]],[v[0],v[1]],'r',linewidth=2)
plt.xlabel(r'$y$')
plt.axis('square')
plt.ylabel(r'$\^y$')
plt.grid()
plt.title('LLS-test')
plt.tight_layout()
plt.savefig('./LLS-yhat_vs_y.png')
plt.show()

#%% Local liner regression
# k = 10
# w_hat_local_steep=[]
# y_hat_tr_local_steep_normal = []
# y_hat_te_local_steep_normal = []
# for (indx,item) in len(X_te_norm):
#   #  print ("---------------->>>>>>>>>>>",indx)
#     break
#     for i in (X_tr_norm.shape):
#         dis_te_tr = [(X_tr_norm - item)]
#     Dis = np.sort(dis_te_tr,axis=1)
#     idx_2 = Dis[:k]
#     X_tr_local = X_tr_norm[idx_2,:]
#     y_tr_local = y_tr_norm[idx_2]
#     Lr = Linear_reg(X_tr_local,y_tr_local) 
#     w_hat_local_steep.append(Lr.Steepest_descent(1000))
#     #_hat_tr_local_steep_normal.append(X_tr_local@w_hat_local_steep[indx])
#     y_hat_te_local_steep_normal.append(X_te_norm[indx,:]@w_hat_local_steep[indx])
    
# y_hat_te_local_steep_normal = np.array(y_hat_te_local_steep_normal)
#%% statistics of the errors LLS
print(10*'-','statistics of the errors_LLS',10*'-')
E_tr_max=E_tr.max()
E_tr_min=E_tr.min()
E_tr_mu=E_tr.mean()
E_tr_sig=E_tr.std()
E_tr_MSE=np.mean(E_tr**2)
R2_tr=1-E_tr_MSE/(np.std(y_tr)**2)
c_tr=np.mean((y_tr-y_tr.mean())*(y_hat_tr-y_hat_tr.mean()))/(y_tr.std()*y_hat_tr.std())
E_te_max=E_te.max()
E_te_min=E_te.min()
E_te_mu=E_te.mean()
E_te_sig=E_te.std()
E_te_MSE=np.mean(E_te**2)
R2_te=1-E_te_MSE/(np.std(y_te)**2)
c_te=np.mean((y_te-y_te.mean())*(y_hat_te-y_hat_te.mean()))/(y_te.std()*y_hat_te.std())
cols=['min','max','mean','std','MSE','R^2','corr_coeff']
rows=['Training','test']
p=np.array([
    [E_tr_min,E_tr_max,E_tr_mu,E_tr_sig,E_tr_MSE,R2_tr,c_tr],
    [E_te_min,E_te_max,E_te_mu,E_te_sig,E_te_MSE,R2_te,c_te],
            ])

results=pd.DataFrame(p,columns=cols,index=rows)
print(results)

#%% statistics of the errors Steepest_Descent
print(10*'-','statistics of the errors_Steepest Descent',10*'-')
E_tr_max_SD=E_tr_SD.max()
E_tr_min_SD=E_tr_SD.min()
E_tr_mu_SD=E_tr_SD.mean()
E_tr_sig_SD=E_tr_SD.std()
E_tr_MSE_SD=np.mean(E_tr_SD**2)
R2_tr_SD=1-E_tr_MSE_SD/(np.std(y_tr)**2)
c_tr_SD=np.mean((y_tr-y_tr.mean())*(y_hat_tr_SD-y_hat_tr_SD.mean()))/(y_tr.std()*y_hat_tr_SD.std())
E_te_max_SD=E_te_SD.max()
E_te_min_SD=E_te_SD.min()
E_te_mu_SD=E_te_SD.mean()
E_te_sig_SD=E_te_SD.std()
E_te_MSE_SD=np.mean(E_te_SD**2)
R2_te_SD=1-E_te_MSE_SD/(np.std(y_te)**2)
c_te_SD=np.mean((y_te-y_te.mean())*(y_hat_te_SD-y_hat_te_SD.mean()))/(y_te.std()*y_hat_te_SD.std())
cols=['min','max','mean','std','MSE','R^2','corr_coeff']
rows=['Training','test']
p=np.array([
    [E_tr_min_SD,E_tr_max_SD,E_tr_mu_SD,E_tr_sig_SD,E_tr_MSE_SD,R2_tr_SD,c_tr_SD],
    [E_te_min_SD,E_te_max_SD,E_te_mu_SD,E_te_sig_SD,E_te_MSE_SD,R2_te_SD,c_te_SD],
            ])

results=pd.DataFrame(p,columns=cols,index=rows)
print(results)
