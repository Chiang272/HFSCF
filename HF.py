import numpy as np
import pandas as pd
import HFSCF.integrals as integrals

#class HFSCF:
#    def __init__(self, mf, mc = None, opt_einsum = False, aux_basis = None):


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
    print("nelec=",nelec)
    #nelec=5
    nocc = np.ceil(nelec/2)
    nocc = int(nocc)
    print("nocc=",nocc) 
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
    #print("finish")
    #print(v2e_so.shape)
    #print(v2e_ao[0,1,:,:])
    #print(v2e_so[0,1,:,:])
    #print(H1_so[0:nelec,0:nelec])
    E1 = np.trace(H1_so[0:nelec,0:nelec])
    print(E1)

    E2 = 0
    #for m in range(nelec):
    #     for n in range(nelec):
    #         E2 = E2 + v2e_so[m,m,n,n] - v2e_so[m,n,m,n]
    #         #print(E2)
    
    vee = np.einsum('m m n n -> ', v2e_so[:nelec, :nelec, :nelec, :nelec])
    vex = np.einsum('m n m n -> ', v2e_so[:nelec, :nelec, :nelec, :nelec])

    E2 = vee - vex

    E2 = E2 /2 
    print(E2)
    

    
    




    


 






