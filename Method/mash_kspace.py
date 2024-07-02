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



def initElectronic(Nstates, initState, Hij, k):
    
    while(True):
        x = np.random.normal(size=Nstates)
        y = np.random.normal(size=Nstates)
        if (np.argmax(x**2 + y**2) == initState):
            break

    norm = np.sqrt(np.sum(x**2 + y**2))
    ck = (x+1j*y)/norm
    phasek = np.angle(ck[initState])
    ck = ck*np.exp(-1j*phasek)
    _, U = np.linalg.eigh(Hij)
    c = np.conj(U).T @ ck
    return c # in adiabatic basis
 
def Force_on_p(dHij_dq, acst, U):
    # dHij is in the diabatic basis !IMPORTANT
    # <a |dH | a> -->   ∑ij <a | i><i | dH |j><j| a>
    F = np.einsum('j, ijk, k -> i', U[:, acst].conjugate(), dHij_dq + 0j, U[:,acst]).real
    return F

def Force_on_q(dHij_dp, acst, U):
    # dHij is in the diabatic basis !IMPORTANT
    # <a |dH | a> -->   ∑ij <a | i><i | dH |j><j| a>
    F = np.einsum('j, ijk, k -> i', U[:, acst].conjugate(), dHij_dp + 0j, U[:,acst]).real
    return F 

def VelVer(ogdat, acst, dt) : 

    dat = copy.deepcopy(ogdat)

    par =  dat.param
    F_on_p = dat.F_on_p * 1.0
    F_on_q = dat.F_on_q * 1.0

    # half electronic evolution
    dat.ci = dat.ci * np.exp(-1j*dt*dat.E/2.0)
    cD =  dat.U @ dat.ci # to diabatic basis
    # ======= Nuclear Block =================================

    dat.P -= (dat.ω**2 * dat.R + F_on_p) * (dt/2.0)
    dat.R += (dat.P + F_on_q) * dt
    dat.P -= (dat.ω**2 * dat.R + F_on_p) * (dt/2.0)

    #------ Do QM ----------------
    dat.Hij  = par.Hel(dat.R,dat.P) + 0j
    dat.dHij_dp = par.dHel_dp(dat.R,dat.P)
    dat.dHij_dq = par.dHel_dq(dat.R,dat.P)
    #dat.dH0  = par.dHel0(dat.R,dat.P)
    #-----------------------------
    dat.E, dat.U = np.linalg.eigh(dat.Hij) 
    dat.F_on_q = Force_on_q(dat.dHij_dp, acst, dat.U) # force at t2
    dat.F_on_p = Force_on_p(dat.dHij_dq, acst, dat.U) # force at t2
    # ======================================================
    dat.ci = np.conj(dat.U).T @ cD # back to adiabatic basis
    
    # half electronic evolution
    dat.ci = dat.ci * np.exp(-1j*dt*dat.E/2.0)

    return dat

def pop(c): # returns the density matrix estimator (populations and coherences)
    NStates = len(c)
    sumN = np.sum(np.array([1/n for n in range(1,NStates+1)])) # constant based on NStates
    alpha = (NStates - 1)/(sumN - 1) # magnitude scaling
    beta = (1-alpha )/NStates # effective zero-point energy
    prod = np.outer(c,np.conj(c)) 
    return alpha * prod + beta * np.identity(NStates) # works in any basis

def corr (ck, k, Nstates, initState):
    sumN = np.sum(np.array([1/n for n in range(1,Nstates+1)])) # constant based on NStates
    alpha = (Nstates - 1)/(sumN - 1) # magnitude scaling
    #print('alpha=',alpha)
    return alpha*ck[initState]

def checkHop(acst, c): # calculate current active state and store result
    # returns [hop needed?, previous active state, current active state]
    n_max = np.argmax(np.abs(c))
    if(acst != n_max):
        return True, acst, n_max
    return False, acst, acst

def γ_ab (dat, a, b, δ_p, δ_q):
    δ = np.cos(dat.k*dat.n) * δ_q - dat.ω * np.sin(dat.k*dat.n) * δ_p
    
    k = dat.k
    Pn = np.zeros(dat.NStates)
    for i in range(dat.NStates):
        Pn[i] = (dat.ω/np.sqrt(dat.NStates))*np.sum(-np.sin(k*i)*dat.R+np.cos(k*i)*dat.P/dat.ω)

    a_ = np.sum((δ/np.linalg.norm(δ))**2)/2
    b_ = -1*np.sum(δ*Pn)/np.linalg.norm(δ)
    c_ = dat.E[b] - dat.E[a]

    if b_ <= 0:
        return (-b_ - np.sqrt(b_**2 - 4*a_*c_))/(2*a_), δ, Pn
    else:
        return (-b_ + np.sqrt(b_**2 - 4*a_*c_))/(2*a_), δ, Pn

def hop(dat, a, b):
    
    if a != b:
        # a is previous active state, b is current active state
        P = dat.P # momentum rescaled
        R = dat.R
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

        cad = np.einsum('m, mf -> f', dat.ci.conjugate(), dat.U)

        #v2
        fma = np.einsum('f, mf, f -> m', cad, dat.U.conjugate(), rΔEa)
        fmb = np.einsum('f, mf, f -> m', cad, dat.U.conjugate(), rΔEb)
        
        term1_p = np.einsum('n, mnk, k -> m', fma, dat.dHij_dp, Ψa * cad[a].conjugate())
        term2_p = np.einsum('n, mnk, k -> m', fmb, dat.dHij_dp, Ψb * cad[b].conjugate())

        term1_q = np.einsum('n, mnk, k -> m', fma, dat.dHij_dq, Ψa * cad[a].conjugate())
        term2_q = np.einsum('n, mnk, k -> m', fmb, dat.dHij_dq, Ψb * cad[b].conjugate())
        
        δk_p = (term1_p - term2_p).real 
        δk_q = (term1_q - term2_q).real

        # information from real space rescaling
        γ, δ, Pn = γ_ab(dat, a, b, δk_p, δk_q)

        #Project the momentum in real space to the new direction
        Pn_proj = np.dot(Pn,δ) * δ / np.dot(δ, δ) if np.dot(Pn,δ) != 0.0 else P * 0.0

        # #Project the momentum in kspace to the new direction
        # Pk_proj = np.dot(P,δ) * δ / np.dot(δ, δ) if np.dot(P,δ) != 0.0 else P * 0.0

        # #Project the position in kspace to the new direction
        # Rk_proj = np.dot(R,δ) * δ / np.dot(δ, δ) if np.dot(R,δ) != 0.0 else R * 0.0

        #Compute projected norm, which will be useful later
        Pn_proj_norm = np.sqrt(np.dot(Pn_proj,Pn_proj))
        
        if(Pn_proj_norm**2 < 2*ΔE): # rejected hop
            print('rejected hop')
            g = 2*np.dot(Pn,δ)
            P -= g * δk_q / np.dot(δ, δ)
            R += g * δk_p / np.dot(δ, δ)
            accepted = False
        else: # accepted hop
            print('accepted hop')
            R = R + γ * δk_p / np.linalg.norm(δ)
            P = P - γ * δk_q / np.linalg.norm(δ)
            accepted = True
        dat.P = P.real
        dat.R = R.real
        return R.real, P.real, accepted
    return dat.R, dat.P, False

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
    k = parameters.k
    n = np.arange( NStates )
    ω = parameters.w
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
        dat.Hij  = parameters.Hel(dat.R,dat.P)
        dat.dHij_dp = parameters.dHel_dp(dat.R,dat.P)
        dat.dHij_dq = parameters.dHel_dq(dat.R,dat.P)
        #dat.dH0  = parameters.dHel0(dat.R,dat.P)
        dat.k = k
        dat.n = n
        dat.ω = ω
        dat.NStates = NStates
        
        # Call function to initialize mapping variables
        dat.ci = initElectronic(NStates, initState, dat.Hij, k) # np.array([0,1])
        acst = np.argmax(np.abs(dat.ci))
        dat.E, dat.U = np.linalg.eigh(dat.Hij) 
        dat.F_on_q = Force_on_q(dat.dHij_dp, acst, dat.U) # Initial Force
        dat.F_on_p = Force_on_q(dat.dHij_dq, acst, dat.U) # Initial Force
        #----------------------------
        iskip = 0 # please modify
        t0 = time.time()
        for i in range(NSteps): # One trajectory
            print('step:',i)
            # print('R:',dat.R)
            # print('P:',dat.P)
            #------- ESTIMATORS-------------------------------------
            if (i % nskip == 0):
                
                rho_ensemble[iskip] += corr(dat.U @ dat.ci, k, NStates, initState)
                iskip += 1
            #-------------------------------------------------------
            dat0 = vv(dat, acst, dtN)
            
            maxhop = 10
            
            #if(checkHop(acst, dat0.ci)[0]==True):
            if (hop(dat0, acst, checkHop(acst, dat0.ci)[2])[-1]): 
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
                
                R, P, accepted = hop(dat_tm, acst, newacst)
                if accepted:
                    dat_tm.R = R # position update
                    dat_tm.P = P # momentum update
                    acst = newacst
                    
                dat = vv(dat_tm, acst, dtN - tm)
                    
            else:
                dat = dat0
                
                
            
        time_taken = time.time()-t0
        print(f"Time taken: {time_taken} seconds")


    return rho_ensemble

