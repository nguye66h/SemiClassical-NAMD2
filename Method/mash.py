# From https://pubs.aip.org/aip/jcp/article/159/9/094115/2909882/A-multi-state-mapping-approach-to-surface-hopping

import numpy as np
import copy
import time
 

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

# def initElectronic(NStates, initState, Hij): 
#     sumN = np.sum(np.array([1/n for n in range(1,NStates+1)]))
#     alpha = (NStates - 1)/(sumN - 1)
#     beta = (alpha - 1)/NStates
    
#     c = np.sqrt(beta/alpha) * np.ones((NStates), dtype = np.complex_)
#     c[initState] = np.sqrt((1+beta)/alpha)
#     for n in range(NStates):
#         uni = np.random.random()
#         c[n] = c[n] * np.exp(1j*2*np.pi*uni)
#     E, U = np.linalg.eigh(Hij)
#     c = np.conj(U).T @ c
#     return c



def initElectronic(Nstates, initState, Hij):

    k = np.arange(0,2*np.pi,2*np.pi/(Nstates))
    
    # while(True):
    #     x = np.random.normal(size=Nstates)
    #     y = np.random.normal(size=Nstates)
    #     if (np.argmax(x**2 + y**2) == initState):
    #         break

    x = np.loadtxt('x.txt')
    y = np.loadtxt('y.txt')

    norm = np.sqrt(np.sum(x**2 + y**2))
    ck = (x+1j*y)/norm
    phasek = np.angle(ck[initState])
    ck = ck*np.exp(-1j*phasek)
    cn = np.einsum('kn,k->n',np.exp(np.outer(-1j*k,np.arange(Nstates))),ck)/np.sqrt(Nstates)

    _, U = np.linalg.eigh(Hij)
    c = np.conj(U).T @ cn
    return c # in adiabatic basis
 
def Force(dHij, dH0, acst, U):

    # dHij is in the diabatic basis !IMPORTANT
    F = -dH0
    # <a |dH | a> -->   ∑ij <a | i><i | dH |j><j| a>
    F -= np.einsum('j, ijk, k -> i', U[:, acst].conjugate(), dHij + 0j, U[:,acst]).real

    return F

def VelVer(ogdat, acst, dt) : 

    dat = copy.deepcopy(ogdat)

    # print("Rn", dat.R)
    # print("Pn", dat.P)

    par =  dat.param
    v = dat.P/par.M
    F1 = dat.F1 * 1.0

    # half electronic evolution
    dat.ci = dat.ci * np.exp(-1j*dt*dat.E/2.0)
    cD =  dat.U @ dat.ci # to diabatic basis
    # ======= Nuclear Block =================================
    dat.R += v * dt + 0.5 * F1 * dt ** 2 / par.M
    
    #------ Do QM ----------------
    dat.Hij  = par.Hel(dat.R) + 0j
    dat.dHij = par.dHel(dat.R)
    dat.dH0  = par.dHel0(dat.R)
    #-----------------------------
    dat.E, dat.U = np.linalg.eigh(dat.Hij) 
    F2 = Force(dat.dHij, dat.dH0, acst, dat.U) # force at t2
    v += 0.5 * (F1 + F2) * dt / par.M
    dat.F1 = F2 * 1.0
    dat.P = v * par.M
    # ======================================================
    dat.ci = np.conj(dat.U).T @ cD # back to adiabatic basis
    
    # half electronic evolution
    dat.ci = dat.ci * np.exp(-1j*dt*dat.E/2.0)

    return dat


# def VelVer(ogdat, acst, dt) : 

#     dat = copy.deepcopy(ogdat)

#     print("Rn", dat.R)
#     print("Pn", dat.P)

#     par =  dat.param
#     F1 = dat.F1 * 1.0

#     # half electronic evolution
#     dat.ci = dat.ci * np.exp(-1j*dt*dat.E/2.0)
#     cD =  dat.U @ dat.ci # to diabatic basis
#     # ======= Nuclear Block =================================
#     dat.P += (-dat.dH0 + F1) * (dt)
#     dat.R += dat.P * dt
#     print('F1:',F1)
#     print('U',dat.U)
#     print('E',dat.E)
    
#     #------ Do QM ----------------
#     dat.Hij  = par.Hel(dat.R) + 0j
#     dat.dHij = par.dHel(dat.R)
#     dat.dH0  = par.dHel0(dat.R)
#     #-----------------------------
#     dat.E, dat.U = np.linalg.eigh(dat.Hij) 
#     print("E", dat.E)
#     F2 = Force(dat.dHij, dat.dH0, acst, dat.U) # force at t2
#     #dat.P += F2 * (dt/2)
#     dat.F1 = F2 * 1.0
#     # ======================================================
#     dat.ci = np.conj(dat.U).T @ cD # back to adiabatic basis
    
#     # half electronic evolution
#     dat.ci = dat.ci * np.exp(-1j*dt*dat.E/2.0)

#     return dat

def pop(c): # returns the density matrix estimator (populations and coherences)
    NStates = len(c)
    sumN = np.sum(np.array([1/n for n in range(1,NStates+1)])) # constant based on NStates
    alpha = (NStates - 1)/(sumN - 1) # magnitude scaling
    beta = (1-alpha )/NStates # effective zero-point energy
    prod = np.outer(c,np.conj(c)) 
    return alpha * prod + beta * np.identity(NStates) # works in any basis

def corr (ci, k, Nstates, initState):
    sumN = np.sum(np.array([1/n for n in range(1,Nstates+1)])) # constant based on NStates
    alpha = (Nstates - 1)/(sumN - 1) # magnitude scaling
    #print('alpha=',alpha)
    ## this ci is a diabat in real space 
    ck = np.einsum('kn,n->k',np.exp(np.outer(1j*k,np.arange(Nstates))),ci)/np.sqrt(Nstates)
    return alpha*ck[initState]

def checkHop(acst, c): # calculate current active state and store result
    # returns [hop needed?, previous active state, current active state]
    n_max = np.argmax(np.abs(c))
    if(acst != n_max):
        return True, acst, n_max
    return False, acst, acst


def hop(dat, a, b):
    
    if a != b:
        # a is previous active state, b is current active state
        P = dat.P/np.sqrt(dat.param.M) # momentum rescaled
        ΔE = np.real(dat.E[b] - dat.E[a]) 
        
        if (np.abs(ΔE) < 1E-13):
            print("Trivial Crossing")
            return dat.P*1.0, True
        # dij is nonadiabatic coupling
        # <i | d/dR | j> = < i |dH | j> / (Ej - Ei)
        
        Ψa = dat.U[:,a]
        Ψb = dat.U[:,b]
        
        # # direction -> 1/√m ∑f Re (c[f] d [f,a] c[a] - c[f] d [f,b] c[b])  # c[f] = ∑m <m | Ψf> 
        # #            =Re ( 1/√m ∑f ∑nm Ψ[m ,f]^ . (<m | dH/dRk | n> ) . Ψ[n ,a] /(E[a]-E[f])
        j = np.arange(len(dat.E))
        ΔEa, ΔEb = (dat.E[a] - dat.E), (dat.E[b] - dat.E)
        ΔEa[a], ΔEb[b] = 1.0, 1.0 # just to ignore error message
        rΔEa, rΔEb = (a != j)/ΔEa, (b != j)/ΔEb
   
        #v2
        fma = np.einsum('mf, f -> m', dat.U.conjugate(), rΔEa)
        fmb = np.einsum('mf, f -> m', dat.U.conjugate(), rΔEb)
        
        term1 = np.einsum('n, mnk, k -> m', fma, dat.dHij, Ψa)
        term2 = np.einsum('n, mnk, k -> m', fmb, dat.dHij, Ψb)
        
        δk = (term1 - term2).real * 1/np.sqrt(dat.param.M) 

        #Project the momentum to the new direction
        P_proj = np.dot(P,δk) * δk / np.dot(δk, δk) if np.dot(P,δk) != 0.0 else P * 0.0

        #print('np.dot(δk, δk) ',np.dot(δk, δk) )
        #Compute the orthogonal momentum
        P_orth = P - P_proj # orthogonal P

        #Compute projected norm, which will be useful later
        P_proj_norm = np.sqrt(np.dot(P_proj,P_proj))
        
        #Compute the total kinetic energy in the projected direction

        if(P_proj_norm**2 < 2*ΔE): # rejected hop
            print("Rejected Hop")
            P_proj = -P_proj # reverse projected momentum
            P = P_orth + P_proj
            accepted = False
        else: # accepted hop
            print("Accepted Hop")
            P_proj = np.sqrt(P_proj_norm**2 - 2*ΔE)/P_proj_norm * P_proj if P_proj_norm != 0 else P * 0.0 #re-scale the projected momentum
            P = P_orth + P_proj
            accepted = True
        P *= np.sqrt(dat.param.M)
        
        dat.P = P.real
        return P.real, accepted
    return dat.P, False

def runTraj(parameters):
    #------- Seed --------------------
    try:
        np.random.seed(parameters.SEED)
    except:
        pass
    #------------------------------------
    ## Parameters -------------
    NSteps = parameters.NSteps
    NTraj = parameters.NTraj
    NStates = parameters.NStates
    initState = parameters.initState # intial state
    nskip = parameters.nskip
    dtN   = parameters.dtN
    k = np.arange(0,2*np.pi,2*np.pi/(NStates))
    #---------------------------
    if NSteps%nskip == 0:
        pl = 0
    else :
        pl = 1
    rho_ensemble = np.zeros((NSteps//nskip + pl), dtype=complex)
    # Ensemble
    for itraj in range(NTraj): 
        # Trajectory data
        dat = Bunch(param =  parameters )
        dat.R, dat.P = parameters.initR()
        
        # set propagator
        vv  = VelVer



        #----- Initial QM --------
        dat.Hij  = parameters.Hel(dat.R)
        dat.dHij = parameters.dHel(dat.R)
        dat.dH0  = parameters.dHel0(dat.R)
        
        
        # Call function to initialize mapping variables
        dat.ci = initElectronic(NStates, initState, dat.Hij) # np.array([0,1])
        acst = np.argmax(np.abs(dat.ci))
        dat.E, dat.U = np.linalg.eigh(dat.Hij) 
        dat.F1 = Force(dat.dHij, dat.dH0, acst, dat.U) # Initial Force
        #----------------------------
        iskip = 0 # please modify
        t0 = time.time()
        for i in range(NSteps): # One trajectory
            # eph = 0.5 * np.sum(dat.P**2 / parameters.M + (parameters.w * dat.R)**2 )
            
            # if i==0:
            #     np.savetxt("E_traj_mash.txt", np.column_stack( (dtN*i,dat.E[0],dat.E[1],dat.E[acst],eph) ) )
            # else:
            #     with open("E_traj_mash.txt", "a") as f:
            #         np.savetxt(f, np.column_stack( (dtN*i,dat.E[0],dat.E[1],dat.E[acst],eph) ) )
            #------- ESTIMATORS-------------------------------------
            if (i % nskip == 0):
                
                rho_ensemble[iskip] += corr(dat.U @ dat.ci, k, NStates, initState)
                iskip += 1
            #-------------------------------------------------------
            dat0 = vv(dat, acst, dtN)
            
            maxhop = 10
            
            #if(checkHop(acst, dat0.ci)[0]==True):
            if (hop(dat0, acst, checkHop(acst, dat0.ci)[2])[1]): 
                newacst = checkHop(acst, dat0.ci)[2]
                # lets find the bisecting point
                tL, tR = 0, dtN
                for _ in range(maxhop):
                    tm = (tL + tR)/2
                    dat_tm = vv(dat, acst, tm)
                    if checkHop(acst, dat_tm.ci)[0]:
                        tL = tm
                    else:
                        tR = tm
                
                P, accepted = hop(dat_tm, acst, newacst)
                if accepted:
                    dat_tm.P = P # momentum update
                    acst = newacst
                    
                dat = vv(dat_tm, acst, dtN - tm)
                    
            else:
                dat = dat0
                
                
            
        time_taken = time.time()-t0
        print(f"Time taken: {time_taken} seconds")


    return rho_ensemble

