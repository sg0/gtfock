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
#if defined(USE_ELEMENTAL)
#else
    int block[2];
#endif
    
    // create GA for dynamic scheduler
    int nprow = pfock->nprow;
    int npcol = pfock->npcol;
    dims[0] = nprow;
    dims[1] = npcol;
#if defined(USE_ELEMENTAL)
#else
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
#endif
#if defined(USE_ELEMENTAL)
    printf ("init_taskq: Creating %d (h) x %d (w) GA...\n", dims[0], dims[1]);
    //ElGlobalArraysCreate_i( eliga, 2, dims, "array taskid", &pfock->ga_taskid );
    ElInt length = dims[0] * dims[1]; // ReadInc only supports 1-D GA
    ElGlobalArraysCreate_i( eliga, 1, &length, "array taskid", &pfock->ga_taskid );
#else
    pfock->ga_taskid =
        NGA_Create_irreg(C_INT, 2, dims, "array taskid", block, map);
    if (0 == pfock->ga_taskid) {
        return -1;
    }
    PFOCK_FREE(map);
#endif
    
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
    ElInt length, ndim;
    ElGlobalArraysInquire_i( eliga, pfock->ga_taskid, &ndim, &length );
    ElInt pos = idx[0] * length + idx[1];	
    ElGlobalArraysReadIncrement_i( eliga, pfock->ga_taskid, 
                                   &pos, ntasks, &nxtask);
#else
    nxtask = NGA_Read_inc(pfock->ga_taskid, idx, ntasks);   
#endif
    return nxtask;
}
