#include <stdlib.h>
#include <stdio.h>

#if defined(USE_ELEMENTAL)
#include <elem.h>
#else
#include <ga.h>
#endif

#include "config.h"
#include "taskq.h"


int init_taskq(PFock_t pfock)
{
    int dims[2];
    int block[2];
    
    // create GA for dynamic scheduler
    int nprow = pfock->nprow;
    int npcol = pfock->npcol;
    dims[0] = nprow;
    dims[1] = npcol;
    block[0] = nprow;
    block[1] = npcol;    
    int *map = (int *)PFOCK_MALLOC(sizeof(int) * (nprow + npcol));
    if (NULL == map) {
        return -1;
    }    
    for (int i = 0; i < pfock->nprow; i++) {
        map[i] = i;
    }
    for (int i = 0; i < npcol; i++) {
        map[i + nprow] = i;
    }
#if defined(USE_ELEMENTAL)
    int nga;
    ElGlobalArraysCreate_d( eldga, 0, 2, dims, "array taskid", &nga);
    pfock->ga_taskid = nga;
#else
    pfock->ga_taskid =
        NGA_Create_irreg(C_INT, 2, dims, "array taskid", block, map);
    if (0 == pfock->ga_taskid) {
        return -1;
    }
#endif
    PFOCK_FREE(map);
    
    return 0;
}


void clean_taskq(PFock_t pfock)
{
#if defined(USE_ELEMENTAL)
    ElGlobalArraysDestroy_i( eliga, pfock->ga_taskid );
#else
    GA_Destroy(pfock->ga_taskid);
#endif
}


void reset_taskq(PFock_t pfock)
{
    int izero = 0;   
#if defined(USE_ELEMENTAL)
    ElGlobalArraysFill_i( eliga, pfock->ga_taskid, &izero);
#else   
    GA_Fill(pfock->ga_taskid, &izero);
#endif
}


int taskq_next(PFock_t pfock, int myrow, int mycol, int ntasks)
{
    int idx[2];

    idx[0] = myrow;
    idx[1] = mycol;
    int nxtask;
#if defined(USE_ELEMENTAL)
    ElGlobalArraysReadIncrement_i( eliga, pfock->ga_taskid, 
                                   2, idx, ntasks, &nxtask);
#else
    nxtask = NGA_Read_inc(pfock->ga_taskid, idx, ntasks);   
#endif
    return nxtask;
}
