#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <unistd.h>

#include <sys/time.h>

#include "config.h"
#include "taskq.h"
#include "fock_buf.h"

#if defined(USE_ELEMENTAL)
#include <elem.h>
#else
#include <ga.h>
#include <macdecls.h>
#endif

void load_local_bufD(PFock_t pfock)
{
    int *loadrow = pfock->loadrow;
    int *loadcol = pfock->loadcol;
    int sizerow = pfock->sizeloadrow;
    int sizecol = pfock->sizeloadcol;

    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    int lo[2];
    int hi[2];
    int ldD;    
    lo[0] = myrank;
    hi[0] = myrank;
    lo[1] = 0;
    for (int i = 0; i < pfock->num_dmat2; i++) {
    #ifdef GA_NB
    #if defined(USE_ELEMENTAL)
        typedef int ga_nbhdl_t;
    #endif
        ga_nbhdl_t nbnb;
    #endif
        // load local buffers
        double *D1;
        double *D2;
        double *D3;
        hi[1] = pfock->sizeX1 - 1;
#if defined(USE_ELEMENTAL)
	lo[0] = 0;
	hi[0] = 0;
	lo[1] = 0;
        ElGlobalArraysAccess_d( eldga, pfock->ga_D1[i], lo, hi, (void **)&D1, &ldD );
        hi[1] = pfock->sizeX2 - 1;
        ElGlobalArraysAccess_d( eldga, pfock->ga_D2[i], lo, hi, (void **)&D2, &ldD );
        hi[1] = pfock->sizeX3 - 1;
        ElGlobalArraysAccess_d( eldga, pfock->ga_D3[i], lo, hi, (void **)&D3, &ldD );
#else
        NGA_Access(pfock->ga_D1[i], lo, hi, &D1, &ldD);
        hi[1] = pfock->sizeX2 - 1;
        NGA_Access(pfock->ga_D2[i], lo, hi, &D2, &ldD);
        hi[1] = pfock->sizeX3 - 1;
        NGA_Access(pfock->ga_D3[i], lo, hi, &D3, &ldD);
#endif
        int ldD1 = pfock->ldX1;
        int ldD2 = pfock->ldX2;
        int ldD3 = pfock->ldX3;     
        // update D1
        lo[0] = pfock->sfunc_row;
        hi[0] = pfock->efunc_row;
        for (int A = 0; A < sizerow; A++) {
            lo[1] = loadrow[PLEN * A + P_LO];
            hi[1] = loadrow[PLEN * A + P_HI];
            int posrow = loadrow[PLEN * A + P_W];
        #ifdef GA_NB
        #if defined(USE_ELEMENTAL)
            ElGlobalArraysNBGet_d( eldga, pfock->ga_D[i], lo, hi, &(D1[posrow]), &ldD1, &nbnb );
        #else   
            NGA_NbGet(pfock->ga_D[i], lo, hi, &(D1[posrow]), &ldD1, &nbnb);
        #endif
        #else
        #if defined(USE_ELEMENTAL)
            ElGlobalArraysGet_d( eldga, pfock->ga_D[i], lo, hi, &(D1[posrow]), &ldD1 );
        #else 
            NGA_Get(pfock->ga_D[i], lo, hi, &(D1[posrow]), &ldD1);
        #endif
        #endif
        }
        // update D2
        lo[0] = pfock->sfunc_col;
        hi[0] = pfock->efunc_col;
        for (int B = 0; B < sizecol; B++) {
            lo[1] = loadcol[PLEN * B + P_LO];
            hi[1] = loadcol[PLEN * B + P_HI];
            int poscol = loadcol[PLEN * B + P_W];
        #ifdef GA_NB   
        #if defined(USE_ELEMENTAL)
            ElGlobalArraysNBGet_d( eldga, pfock->ga_D[i], lo, hi, &(D2[poscol]), &ldD2, &nbnb );
        #else   
            NGA_NbGet(pfock->ga_D[i], lo, hi, &(D2[poscol]), &ldD2, &nbnb);
        #endif
        #else
        #if defined(USE_ELEMENTAL)
            ElGlobalArraysNBGet_d( eldga, pfock->ga_D[i], lo, hi, &(D2[poscol]), &ldD2 );
        #else   
            NGA_Get(pfock->ga_D[i], lo, hi, &(D2[poscol]), &ldD2);
        #endif
        #endif
        }
        // update D3
        for (int A = 0; A < sizerow; A++) {
            lo[0] = loadrow[PLEN * A + P_LO];
            hi[0] = loadrow[PLEN * A + P_HI];
            int posrow = loadrow[PLEN * A + P_W];
            for (int B = 0; B < sizecol; B++) {
                lo[1] = loadcol[PLEN * B + P_LO];
                hi[1] = loadcol[PLEN * B + P_HI];
                int poscol = loadcol[PLEN * B + P_W];
            #ifdef GA_NB
            #if defined(USE_ELEMENTAL)
                ElGlobalArraysNBGet_d( eldga, pfock->ga_D[i], lo, hi, 
                                       &(D3[posrow * ldD3 + poscol]), &ldD3, &nbnb );
            #else   
                NGA_NbGet(pfock->ga_D[i], lo, hi,
                          &(D3[posrow * ldD3 + poscol]), &ldD3, &nbnb);
            #endif
            #else
            #if defined(USE_ELEMENTAL)
                ElGlobalArraysGet_d( eldga, pfock->ga_D[i], lo, hi, 
                                     &(D3[posrow * ldD3 + poscol]), &ldD3 );
            #else   
                NGA_Get(pfock->ga_D[i], lo, hi,
                        &(D3[posrow * ldD3 + poscol]), &ldD3);        
            #endif
            #endif
            }
        }
    #ifdef GA_NB            
    #if defined(USE_ELEMENTAL)
        ElGlobalArraysNBWait_d( eldga, &nbnb );
    #else
        NGA_NbWait (&nbnb);
    #endif
    #endif
    #if defined(USE_ELEMENTAL)
    #else
        // release update
        lo[0] = myrank;
        hi[0] = myrank;
        lo[1] = 0;
        hi[1] = pfock->sizeX1 - 1;
        NGA_Release_update(pfock->ga_D1[i], lo, hi);
        hi[1] = pfock->sizeX2 - 1;
        NGA_Release_update(pfock->ga_D2[i], lo, hi);
        hi[1] = pfock->sizeX3 - 1;
        NGA_Release_update(pfock->ga_D3[i], lo, hi);
    #endif
    }
}


void store_local_bufF(PFock_t pfock)
{
    int *loadrow = pfock->loadrow;
    int *loadcol = pfock->loadcol;
    int sizerow = pfock->sizeloadrow;
    int sizecol = pfock->sizeloadcol;
    int myrank;
    MPI_Comm_rank (MPI_COMM_WORLD, &myrank);
    int lo[2];
    int hi[2];
    int ldF;
    int *ga_J = pfock->ga_F;
#ifdef __SCF__
    int *ga_K = pfock->ga_F;
#else
    int *ga_K = pfock->ga_K;
#endif
    lo[0] = myrank;
    hi[0] = myrank;
    lo[1] = 0;
    for (int i = 0; i < pfock->num_dmat2; i++) {
    #ifdef GA_NB    
    #if defined(USE_ELEMENTAL)
        typedef int ga_nbhdl_t;
    #endif
        ga_nbhdl_t nbnb;
    #endif
        // local buffers
        double *F1;
        double *F2;
        double *F3;
        hi[1] = pfock->sizeX1 - 1;
#if defined(USE_ELEMENTAL)
        lo[0] = 0; 
        lo[1] = 0;
        hi[0] = 0;
        ElGlobalArraysAccess_d( eldga, pfock->ga_F1[i], lo, hi, (void **)&F1, &ldF );
        ElGlobalArraysAccess_d( eldga, pfock->ga_F2[i], lo, hi, (void **)&F2, &ldF );
        ElGlobalArraysAccess_d( eldga, pfock->ga_F3[i], lo, hi, (void **)&F3, &ldF );
#else
        NGA_Access(pfock->ga_F1[i], lo, hi, &F1, &ldF);
        lo[1] = 0;
        hi[1] = pfock->sizeX2 - 1;
        NGA_Access(pfock->ga_F2[i], lo, hi, &F2, &ldF);
        lo[1] = 0;
        hi[1] = pfock->sizeX3 - 1;
        NGA_Access(pfock->ga_F3[i], lo, hi, &F3, &ldF);
#endif
        int ldF1 = pfock->ldX1;
        int ldF2 = pfock->ldX2;
        int ldF3 = pfock->ldX3;    
        // update F1
        double done = 1.0;
        lo[0] = pfock->sfunc_row;
        hi[0] = pfock->efunc_row;
        for (int A = 0; A < sizerow; A++) {
            lo[1] = loadrow[PLEN * A + P_LO];
            hi[1] = loadrow[PLEN * A + P_HI];
            int posrow = loadrow[PLEN * A + P_W];
        #ifdef GA_NB
        #if defined(USE_ELEMENTAL)
            ElGlobalArraysNBAccumulate_d( eldga, ga_J[i], lo, hi, 
                                   &(F1[posrow]), &ldF1, &done, &nbnb );
        #else
            NGA_NbAcc(ga_J[i], lo, hi, &(F1[posrow]),
                      &ldF1, &done, &nbnb);   
        #endif 
        #else
        #if defined(USE_ELEMENTAL)
            ElGlobalArraysAcc_d( eldga, ga_J[i], lo, hi, 
                                 &(F1[posrow]), &ldF1, &done );
        #else
            NGA_Acc(ga_J[i], lo, hi, &(F1[posrow]), &ldF1, &done);
        #endif
        #endif
        }

        // update F2
        lo[0] = pfock->sfunc_col;
        hi[0] = pfock->efunc_col;
        for (int B = 0; B < sizecol; B++) {
            lo[1] = loadcol[PLEN * B + P_LO];
            hi[1] = loadcol[PLEN * B + P_HI];
            int poscol = loadcol[PLEN * B + P_W];
        #ifdef GA_NB
        #if defined(USE_ELEMENTAL)
            ElGlobalArraysNBAccumulate_d( eldga, ga_J[i], lo, hi, 
                              &(F2[poscol]), &ldF2, &done, &nbnb );
        #else
            NGA_NbAcc(ga_J[i], lo, hi, &(F2[poscol]),
                      &ldF2, &done, &nbnb);
        #endif
        #else
        #if defined(USE_ELEMENTAL)
            ElGlobalArraysAcc_d( eldga, ga_J[i], lo, hi, 
                              &(F2[poscol]), &ldF2, &done );
        #else
            NGA_Acc(ga_J[i], lo, hi, &(F2[poscol]), &ldF2, &done);
        #endif
        #endif
        }

        // update F3
        for (int A = 0; A < sizerow; A++) {
            lo[0] = loadrow[PLEN * A + P_LO];
            hi[0] = loadrow[PLEN * A + P_HI];
            int posrow = loadrow[PLEN * A + P_W];
            for (int B = 0; B < sizecol; B++) {
                lo[1] = loadcol[PLEN * B + P_LO];
                hi[1] = loadcol[PLEN * B + P_HI];
                int poscol = loadcol[PLEN * B + P_W];
            #ifdef GA_NB
            #if defined(USE_ELEMENTAL)
                ElGlobalArraysNBAccumulate_d( eldga, ga_K[i], lo, hi, 
                                       &(F3[posrow * ldF3 + poscol]), &ldF3, &done, &nbnb );
            #else
                NGA_NbAcc(ga_K[i], lo, hi, 
                          &(F3[posrow * ldF3 + poscol]), &ldF3, &done, &nbnb);
            #endif
            #else
            #if defined(USE_ELEMENTAL)
                ElGlobalArraysAcc_d( eldga, ga_K[i], lo, hi, 
                              &(F3[posrow * ldF3 + poscol]), &ldF3, &done );
            #else
                NGA_Acc(ga_K[i], lo, hi, 
                        &(F3[posrow * ldF3 + poscol]), &ldF3, &done);        
            #endif
            #endif
            }
        }
    #ifdef GA_NB
    #if defined(USE_ELEMENTAL)
        ElGlobalArraysNBWait_d( eldga, &nbnb );
    #else
        NGA_NbWait(&nbnb);
    #endif
    #endif
    #if defined(USE_ELEMENTAL)
    #else
        // update release
        lo[0] = myrank;
        hi[0] = myrank;
        lo[1] = 0;
        hi[1] = pfock->sizeX1 - 1;
        NGA_Release(pfock->ga_F1[i], lo, hi);
        lo[1] = 0;
        hi[1] = pfock->sizeX2 - 1;
        NGA_Release(pfock->ga_F2[i], lo, hi);
        lo[1] = 0;
        hi[1] = pfock->sizeX3 - 1;
        NGA_Release(pfock->ga_F3[i], lo, hi);
     #endif
    }
#if defined(USE_ELEMENTAL)
    ElGlobalArraysSync_d( eldga );
#else
    GA_Sync();
#endif
}


void compute_FD_ptr(PFock_t pfock, int startM, int endM,
                    int *ptrrow, int *rowsize)
{
    for (int A = 0; A < pfock->nshells; A++) {
        ptrrow[A] = -1;
    }    
    // init row pointers
    for (int A = startM; A <= endM; A++) {
        int start = pfock->shellptr[A];
        int end = pfock->shellptr[A + 1]; 
        for (int i = start; i < end; i++) {
            int B = pfock->shellid[i];
            ptrrow[B] = 1;
        }
    }
    for (int i = 0; i < pfock->natoms; i++)
    {
        int start = pfock->s_startind[i];
        int end = pfock->s_startind[i + 1];
        int flag = -1;
        for (int A = start; A < end; A++)
        {
            if (ptrrow[A] != -1)
                flag = 1;
        }
        for (int A = start; A < end; A++)
        {
            ptrrow[A] = flag;
        }
    }
    *rowsize = 0;
    for (int A = 0; A < pfock->nshells; A++)
    {
        if (ptrrow[A] == 1)
        {
            ptrrow[A] = *rowsize;           
            *rowsize += pfock->f_startind[A + 1] - pfock->f_startind[A];
        }
    }
}


void init_FD_load(PFock_t pfock, int *ptrrow,
                  int **loadrow, int *loadsize)
{    
    int loadcount = 0;
    for (int A = 0; A < pfock->nshells; A++) {
        if (ptrrow[A] != -1) {
            while (A < pfock->nshells && ptrrow[A] != -1) {
                A++;
            }           
            loadcount++;
        }
    }
    *loadrow = (int *)PFOCK_MALLOC(sizeof(int) * PLEN * loadcount);
    assert(NULL != *loadrow);
    *loadsize = loadcount;
    
    loadcount = 0;
    for (int A = 0; A < pfock->nshells; A++) {
        int idx = ptrrow[A];
        if (idx != -1) {
            int lo = pfock->f_startind[A];
            while (A < pfock->nshells && ptrrow[A] != -1) {
                A++;
            }           
            int hi = pfock->f_startind[A] - 1;
            (*loadrow)[loadcount * PLEN + P_LO] = lo;
            (*loadrow)[loadcount * PLEN + P_HI] = hi;
            (*loadrow)[loadcount * PLEN + P_W] = idx;
            loadcount++;
        }
    }
}
