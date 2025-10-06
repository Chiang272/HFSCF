import numpy as np
import pandas as pd
import HFSCF.integrals as integrals
import math

#class HFSCF:
#    def __init__(self, mf, mc = None, opt_einsum = False, aux_basis = None):



def find_occ(config,nso):
    list=[]
    for i in range(nso):
        if config & (1<<i):
            list.append(i)
    return list

def kernel(mol):
    print("runing HFSCF...")
    
    natm = mol.natm
    nelec = mol.nelectron
    # Obtain the number of atomic orbitals in the basis set
    nao = mol.nao_nr()

    S_ao = mol.intor('cint1e_ovlp_sph')
    T_ao = mol.intor('cint1e_kin_sph')
    V_ao = mol.intor('cint1e_nuc_sph')
    H1_ao = T_ao + V_ao

    v2e_ao = mol.intor('cint2e_sph').reshape((nao,)*4)
    '''
    #test szabo####
    S_ao = np.array([[1.0,0.4508],
                     [0.4508,1.0]])
    H1_ao = np.array([[-2.6527,-1.3472],
                      [-1.3472,-1.7318]])
    v2e_ao = np.zeros([2,2,2,2])

    v2e_ao[0,0,0,0] = 1.3072
    v2e_ao[1,0,0,0] = 0.4373
    v2e_ao[1,0,1,0] = 0.1773
    v2e_ao[1,1,0,0] = 0.6057
    v2e_ao[1,1,1,0] = 0.3118
    v2e_ao[1,1,1,1] = 0.7746

    v2e_ao[0,1,0,0] = v2e_ao[1,0,0,0]
    v2e_ao[0,0,1,0] = v2e_ao[1,0,0,0]
    v2e_ao[0,0,0,1] = v2e_ao[1,0,0,0]

    v2e_ao[0,1,1,0] = v2e_ao[1,0,1,0]
    v2e_ao[0,1,0,1] = v2e_ao[1,0,1,0]
    v2e_ao[1,0,0,1] = v2e_ao[1,0,1,0]

    v2e_ao[1,1,0,1] = v2e_ao[1,1,1,0]
    v2e_ao[0,1,1,1] = v2e_ao[1,1,1,0]
    v2e_ao[1,0,1,1] = v2e_ao[1,1,1,0]

    v2e_ao[0,0,1,1] = v2e_ao[1,1,0,0]
    #test szabo###
    '''

    print("Basis set:",mol.basis)
    print("number of atoms:", natm)
    print("number of electrons:", nelec)

    print("computing nuclear repulsion energy...")
    Enn = 0
    for I in range(natm):
        for J in range(natm):
            if I != J:

                vector = mol.atom_coord(I) - mol.atom_coord(J)
                #print(vector)
                dis = np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
                #print(dis)
                Enn += mol.atom_charge(I) * mol.atom_charge(J)/ dis
    Enn = Enn/2
    print("Enn=", Enn)

    print("Construct the S^(-1/2) matrix.")
    #print(S_ao)

    sao_en , sao_evec = np.linalg.eigh(S_ao)
    sao_en_2 = np.diag(sao_en ** (-1/2))
    SS = sao_evec @ (sao_en_2) @ sao_evec.T
    #print(SS)

    print("Construct initial density matrix.")
    F = H1_ao 
    F_tilde = SS @ F @ SS
    F_tilde_en , F_tilde_evec = np.linalg.eigh(F_tilde)

    C = SS @ F_tilde_evec
    C_half = C[:,0:int(nelec/2)]
    D = np.einsum('ui,vi->uv',C_half,C_half) #* 2 
    #print(D*2)

    #SCF interation####
    print("SCF interation...")
    N = nao
    #v2e_ao_2 = np.einsum('uvsl->ulsv',v2e_ao) 
    #count = 1
    E_elec_old = 0
    E_elec = 100

    D_old = np.zeros((len(D),len(D[0])))

    #while (np.abs(E_elec-E_elec_old) > 1e-12 and count < 100):
    #for i in range(1000):
    print("count ||||  delta_energy  ||  delta_D")
    for count in range(100):
        delta_D = D -D_old
        delta_D2 = delta_D * delta_D 
        #print(count)
        #print("delta E =",E_elec-E_elec_old)
        #print("delta D =",np.sqrt(np.sum(delta_D2)))
        #print(str(count)+" ||||  %14.6f  ||  %14.6f"%( (E_elec-E_elec_old),np.sqrt(np.sum(delta_D2))))
        print(str(count)+"|||"+str(E_elec-E_elec_old)+"||" + str(np.sqrt(np.sum(delta_D2))))
        if (np.abs(E_elec-E_elec_old) < 1e-12 and np.sqrt(np.sum(delta_D2)) < 1):
            print("SCF converage")
            break
        else:    
            #print(count)
            count = count + 1
            E_elec_old = E_elec
            D_old = D
            G = 2* np.einsum('ls,uvsl->uv',D,v2e_ao) - np.einsum('ls,ulsv->uv',D,v2e_ao)
            #print(G)

            F = H1_ao + G

            #print(F)

            E_elec = np.einsum('uv,uv',D,H1_ao+F)
            #print(E_elec)

            #E_total = E_elec + Enn
            #print(E_total)

            F_tilde = SS @ F @ SS
            F_tilde_en , F_tilde_evec = np.linalg.eigh(F_tilde)
            C = SS @ F_tilde_evec
            C_half = C[:,0:int(nelec/2)]
            D = np.einsum('ui,vi->uv',C_half,C_half) #* 2 
            #print(D*2)

    E_total = E_elec + Enn

    print("Total Energy = ",E_total)


def FullCI(mf):
    Enn = mf.mol.energy_nuc()
    print("Enn=", Enn)
    mo = mf.mo_coeff
    nao = mf.mol.nao_nr()
    nelec = mf.mol.nelectron

    S_ao = mf.mol.intor('cint1e_ovlp_sph')
    T_ao = mf.mol.intor('cint1e_kin_sph')
    V_ao = mf.mol.intor('cint1e_nuc_sph')
    H1_ao = T_ao + V_ao

    # below is Chemists notation
    v2e_ao = mf.mol.intor('cint2e_sph').reshape((nao,)*4)

    #transfer to spin-orbit basis
    interface=0
    H1_so  = integrals.transform_1e_integrals_so_2(mo, H1_ao)
    #H1_so_2  = integrals.transform_1e_integrals_so(interface,mo, H1_ao)
    #print(H1_so)
    #print(H1_so_2)
    #exit()
    
    #nelec=5
    #nocc = np.ceil(nelec/2)
    #nocc = int(nocc)
    #print("nocc=",nocc) 
    #print("Mo=")
    #print(mo)
    #mo_c = mo[:, :nocc].copy()
    #mo_e = mo[:, nocc:].copy()
    #print("mo_c=")
    #(mo_c)
    #print("mo_e=")
    #print(mo_e)
    #mo=np.diag([1,1])
    #print(mo)
    
    print("transfer 2-e integral in so basis...")
    v2e_so = integrals.transform_2e_integrals_so_2(mo,v2e_ao)
    print("successfully transfer")

    #calculate number of configuration
    nso=len(H1_so)
    print("nelec=",nelec)
    print("nso=",nso)
    #ncon = math.factorial(nso) // (math.factorial(nelec)*math.factorial(nso-nelec))
    #print("ncon=" ,ncon)

    
    #build configuration
    config_old=[]
    for i in range(int(2**(nso))):
        if bin(i).count('1') == nelec:
            config_old.append(i)
    #print(config)
    '''
    config = []
    for i in config_old:
        locate = find_occ(i,nso)
        even = 0
        odd  = 0
        for j in locate:
            if j % 2 ==1:
                odd += 1
            elif j % 2 == 0:
                even += 1
        if even == 2 & odd == 2:
            config.append(i)
    ncon = len(config)
    print(config)
    print("ncon=",ncon)
    '''

    '''
    #build spin restrcit configuration
    config=[] 
    ncon =  nao * nao  # only work for 2 electrons with up and down
    print("nao=", nao)
    for i in range(nao):
        for j in range(nao):
            CI = 2 **(i*2+1) + 2 ** (2*j)
            config.append(CI)
    print(config)
    '''

    #print(config)
    #print(bin(config))
    #exit()
    
    
    
    # only work for 4 electrons with up and down
    config=[] 
    #print("nao=", nao)
    for i in range(nao-1):
        i += 1
        for j in range(i):
            for k in range(nao-1):
                k += 1
                for l in range(k):
                    #if (i != j & k != l):
            
                    CI = 2**(i*2+1)+ 2**(j*2+1) + 2**(2*k) + 2**(2*l)
                    config.append(CI)
    ncon = len(config)
    print(config)
    
    #exit()
    #build Full CI matrix###############
    H_FCI = np.zeros([ncon,ncon])
    
    #diagonal part##############################

    for i in range(ncon): #ncon):
        mo_occ_list = []
        E1=0
        E2=0
        for j in range(nso):
            if config[i]&(1<<j):
                mo_occ_list.append(j)
                E1+= H1_so[j,j]
        
        for k in range(nelec):
            for l in range(nelec):
                m=mo_occ_list[k]
                n=mo_occ_list[l]

                E2+= v2e_so[m,m,n,n]
                E2-= v2e_so[m,n,m,n]
        E2 = E2 /2 

        H_FCI[i,i] += E1 + E2
        #print(bin(config[i]))
        #print(H_FCI[i,i])
    #print("H1_so=")
    #print(H1_so)
    ####offdiagnoal term#############
    H_FCI_off = np.zeros([ncon,ncon])
    test_j = np.zeros(ncon)
    
    #print(H1_so)
    for I in range(ncon):
        #print("I=",I)
        #print(bin(config[I]))
        for J in range(ncon):
            #J = 2
            #print("J=",J)
            if (I>J):
                
                J_occ = find_occ(config[J],nso)

                result= config[I] ^ config[J]
                #print(bin(result))
                ### different by one spin-orbital
                if bin(result).count('1') == 2:
                    test_j[I] = 1
                    #print("found")
                    #print(bin(result))
                    diff_occ = find_occ(result,nso)
                    #(bin(config[I]))
                    #print(bin(config[J]))
                    #(diff_occ)
                    h1_off = H1_so[diff_occ[0],diff_occ[1]]
                    #print("h1_0ff=", h1_off)
                    
                    h2_off = 0
                    for i in range(nelec):
                        m = J_occ[i]
                        h2_off += v2e_so[diff_occ[0],diff_occ[1],m,m]
                        h2_off -= v2e_so[diff_occ[0],m,diff_occ[1],m]
                    #("h1_off=",h1_off)
                    #print("h2_off=",h2_off)
                    H_FCI_off[I,J] = h1_off + h2_off
                    #H_FCI_off[I,J] =0
                    #print("H1_off=",I,"and",J,":",H_FCI_off[I,J])
                ### different by 2 spin-orbital
                elif bin(result).count('1') == 4:
                    #print("doublet")
                    test_j[I] = 2
                    config_J_elec = config[J] & result
                    config_I_elec = config[I] & result
                    J_occ = find_occ(config_J_elec,nso)
                    I_occ = find_occ(config_I_elec,nso)

                    if I_occ[0] % 2 == J_occ[0] % 2:
                        i = J_occ[0]
                        j = J_occ[1]
                        a = I_occ[0] 
                        b = I_occ[1] 
                    else:
                        i = J_occ[0]
                        j = J_occ[1]
                        a = I_occ[1] 
                        b = I_occ[0] 

                    #print("energy=")
                    #print(v2e_so[i,a,j,b] - v2e_so[i,j,a,b])

                    #diff_occ = find_occ(result,nso)
                    #print(diff_occ)
                    H_FCI_off[I,J] = v2e_so[i,a,j,b] - v2e_so[i,j,a,b] #v2e_so[diff_occ[0],diff_occ[2],diff_occ[1],diff_occ[3]] - v2e_so[diff_occ[0],diff_occ[1],diff_occ[2],diff_occ[3]]
    
    print(config[0])
    print(config[10])
    #exit()
    
    result = config[0] ^ config[10]
    print(bin(result))
    config_J_elec = config[0] & result
    config_I_elec = config[10] & result
    J_occ = find_occ(config_J_elec,nso)
    I_occ = find_occ(config_I_elec,nso)



    print(J_occ)
    print(I_occ)

    i = J_occ[0]
    j = J_occ[1]
    a = I_occ[1] 
    b = I_occ[0] 
    print(v2e_so[i,a,j,b] - v2e_so[i,j,a,b])
    print(H_FCI_off[10,0])
    #if H_FCI_off[10,0] < 0:
    #    H_FCI_off = - H_FCI_off

    #H_FCI_off = np.abs(H_FCI_off)
    
    '''
    print(test_j)
    print(H_FCI_off[:,0])

    print("Ewwwwwwwwwwwwwww")

    #confg1 =  3
    #confg2 = 72
    result = config[0] ^ config[5]

    config_J_elec = config[0] & result
    config_I_elec = config[5] & result
    print(bin(config_J_elec))
    print(bin(config_I_elec))
    print("result=",result)
    print("nso=",nso)

    J_occ = find_occ(config_J_elec,nso)
    I_occ = find_occ(config_I_elec,nso)
    print("dif_occ=")
    #diff_occ = [0,1,2,7]
    print(J_occ)
    print(I_occ)

    if I_occ[0] % 2 == J_occ[0] % 2:
        i = J_occ[0]
        j = J_occ[1]
        a = I_occ[0] 
        b = I_occ[1] 
    else:
        i = J_occ[0]
        j = J_occ[1]
        a = I_occ[1] 
        b = I_occ[0] 

    print("energy=")
    print(v2e_so[i,a,j,b] - v2e_so[i,j,a,b])

    exit()
    '''
    #print(H_FCI_off)
    
    H_FCI = H_FCI + H_FCI_off + H_FCI_off.T
    #E_HF = np.diag(H_FCI)
    print("Inhome FCI H:")
    print(H_FCI)
    print("ncon=",ncon)
    print("nso=",nso)
    print("nelec=",nelec)
    print("HF_energy=",H_FCI[0,0]+Enn)
    #print(H_FCI-H_FCI.T)
    # DIAGONALIZE:
    FCI_en, FCI_evec = np.linalg.eigh(H_FCI)

    #print(FCI_evec[:,0].reshape(nao,nao))


    #print("FCI eigenvaluse:")
    #print(FCI_evec)
    
    print("total FCI energy")
    #print(FCI_)
    print(FCI_en[0]+Enn)
    #print(FCI_en+Enn)
    #from pyscf import fci
    #H_fci = fci.direct_spin1.pspace(H1_ao, v2e_ao, nso//2, nelec, np=1225)[1]
    #print(H_fci.shape)
    #e_all, v_all = np.linalg.eigh(H_fci)
    #print("e_all(test)=",e_all[0]+Enn)
    
    
    
    #print(E1)
    #print(E2)

    #E1 = np.trace(H1_so[0:nelec,0:nelec])
    #print(E1)
    #E2 = 0   
    #vee = np.einsum('m m n n -> ', v2e_so[:nelec, :nelec, :nelec, :nelec])
    #vex = np.einsum('m n m n -> ', v2e_so[:nelec, :nelec, :nelec, :nelec])

    #E2 = vee - vex

    #E2 = E2 /2 
    #print(E2)

    return H_FCI
    
    

    




    


 






