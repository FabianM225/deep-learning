import torch
import numpy as np
torch.set_printoptions(precision=8)

def fastnnls(X,Xty,tol,b,PP,device='cpu'):
    # Fast NNLS via active-set method with warm-starting from previous solution PP


    #device = "cpu"
    n = X.shape[1]
    # Tag det gamle aktive sæt, som er de værdier, der ikke er 0.

    if not len(PP):
        P = (torch.empty((n,1), dtype=torch.long,device=device)*np.nan).flatten()
        PP = torch.argwhere(~torch.isnan(P)).flatten()

        Z = torch.arange(1,n+1,dtype=torch.float,device=device)
        ZZ = (Z-1).type(torch.long)

    else:
        P = (torch.empty((n,1), dtype=torch.float,device=device)*np.nan).flatten()
        P[PP] = PP.float()

        Z = torch.arange(1,n+1,dtype=torch.float,device=device)
        Z[PP] = np.nan
        ZZ = torch.argwhere(torch.isnan(P)).flatten()

    x = b.clone().type(torch.double)
    z = b.clone().type(torch.double)
   
    ij = None
    iterOuter = 0

    scale = torch.mean(torch.sum(X*X,axis =1))
    lambda1 = 1e6*scale
    lambda2 = 1e-3*scale
    t = X[:,PP]@x[PP]
    w = Xty +lambda1 - X.T@t - torch.sum(x)*lambda1 - lambda2*x
    iter = 0 
    # ændre tol til tol/np.mean(np.diag(XtX))

    if len(PP) == 0:
        XtX_PP = None 
    else:
        XtX_PP = X[:,PP].T@X[:,PP]

    
    while(len(ZZ)>0 and torch.sum(w[ZZ]>tol)>0 and iterOuter<1000):

        iter = 0
        temp = w[ZZ]
        t = torch.argmax(temp)
        t = ZZ[t]
        #P[t] = t # Add the index to the set of active indices.
        #Z[t] = np.nan # Remove the index from the set of candidate indices.


        #PP = PP.append(t) #torch.argwhere(~torch.isnan(P)).flatten() # Select the active indices.
        PP = torch.cat((PP,torch.tensor([t],dtype=torch.long,device=device)),0)
        ZZ = ZZ[ZZ!=t]  #torch.argwhere(~torch.isnan(Z)).flatten() # Select the candidate indices.

        if XtX_PP is None:
            XtX_PP = X[:,PP].T@X[:,PP]
        else:
            #XtX_PP = XtX_PP.append(X[:,t].T@X[:,PP[:-1]],axis = 1)
            #XtX_PP = XtX_PP.append(X[:,PP].T@X[:,t],axis = 0)

            if len(PP[:-1]) == 1:
                XtX_PP = torch.hstack((XtX_PP[0],X[:,t]@X[:,PP[:-1]]))
                XtX_PP = torch.vstack((XtX_PP,X[:,PP].T@X[:,t]))

            else:

                XtX_PP = torch.hstack((XtX_PP,(X[:,t]@X[:,PP[:-1]]).unsqueeze(1)))
                XtX_PP = torch.vstack((XtX_PP,X[:,PP].T@X[:,t]))


        #lamb_inner =  torch.mean(torch.diag(XtX_PP)) 
        #lamb = lamb_inner*10e9 # ændret fra 1

        activeSetChange = True
        z = z*0
        #z[PP] = torch.linalg.solve(X[:,PP].T@X[:,PP]*SSt+lamb,Xty[PP]+lamb) # Compute the search direction.
        z[PP] = torch.linalg.solve(XtX_PP+lambda1+torch.eye(len(PP), device=device)*lambda2,Xty[PP]+lambda1)


        while (torch.sum(z[PP]<=tol)>0 and iter<500): 


            temp = torch.argwhere(z[PP] <= tol).T[0]

            QQ =PP[temp] # ]temp[(temp.view(1, -1) == temp2.view(-1, 1)).any(dim=0)]


            #QQ = np.intersect1d(temp,temp2) # Select the indices with small search direction and active indices.

            a = x[QQ]/((x[QQ]-z[QQ]))
            a[torch.isnan(a)] = np.inf
            alpha = torch.min(a)

            x = x + alpha*(z-x)

            t1 = torch.argwhere(x[PP]<tol).flatten()
            ij = PP[t1]

            idx_PP = torch.ones_like(PP).bool() if len(ij)==0 else ~torch.eq(PP[:, None], ij).T[0] 

            PP = PP[idx_PP]

            
            ZZ = torch.cat((ZZ,ij),0)



            if XtX_PP is None:
                XtX_PP = X[:,PP].T@X[:,PP]
            else:
                XtX_PP = XtX_PP[idx_PP][:,idx_PP]




            #lamb = torch.mean(torch.diag(XtX_PP))*10e9

            #z[PP] = torch.linalg.solve(X[:,PP].T@X[:,PP]*SSt+lamb,Xty[PP]+lamb) # Compute the search direction.
            z = z*0
            z[PP] = torch.linalg.solve(XtX_PP+lambda1+torch.eye(len(PP), device=device)*lambda2,Xty[PP]+lambda1)



            iter = iter + 1

        
        x = z.clone()
        
        t = X[:,PP]@x[PP]
        w = Xty +lambda1 - X.T@t - torch.sum(x)*lambda1 - lambda2*x 
    
        iterOuter = iterOuter + 1

    
    out = z.clone()
        
    return out,w,PP
