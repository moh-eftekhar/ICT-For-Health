# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 12:33:57 2022

@author: meftekhar
"""
import numpy as np
# np.random.seed (71) # s e t s t h e s e ed us ed t o g e n e r a t e
# # t h e random v a r i a b l e s t o 71
# X=np.random.rand(0,1) # g e n e r a t e s a ma t r i x MxN wi th random v a l u e s
# # u n i f . d i s t r . i n [ 0 , 1 )
# X=np.random.randn(0,1) # g e n e r a t e s a ma t r i x MxN wi th Ga u s s i a n
# # d i s t r i b . random v a l u e s (mean 0 , v a r 1)
# X=np.random.randint(3,size=(0,1)) # g e n e r a t e s a ma t r i x MxN wi t h
# # i n t e g e r random v a l u e s u n i f d i s t r . i n [ 0 ,K−1]
# np.random.shuffle(X) # randomly p e rmu t e s / s h u f f l e s t h e e l eme n t s of
# #x ( i n p l a c e )
Np=5 # Number of rows
Nf=4 # Number of columns
A=np.random.randn(Np,Nf) # Gauss . random ma t r i x A, shape (Np , Nf )
w=np.random.randn(Nf) # Gauss . random v e c t o r w id , shape (Nf , )
y=A@w
ATA=A.T@A # g e n e r a t e AT*A
ATAinv=np.linalg.inv(ATA) # g e n e r a t e (AT*A) * * ( −1 )
ATy=A.T@y # g e n e r a t e AT*y
w_hat=ATAinv@ATy
print (w_hat)