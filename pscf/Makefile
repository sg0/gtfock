include ../make.in

EXEC = scf
SRC = $(wildcard *.c)
OBJS := $(addsuffix .o, $(basename $(SRC)))

ifeq "${SCF_TIMMING}" "1"
CFLAGS += -D__SCF_TIMING__
endif

ifeq "${SCF_OUT}" "1"
CFLAGS += -D__SCF_OUT__
endif

ifeq "${ELEMENTAL}" "1"
INC = -I./                  \
      -I../install          \
      -I${EL_INCDIR}        \
      -I${ERD_INCDIR}       \
      -I${MPI_INCDIR}       \
      -I${SCALAPACK_INCDIR} \
      -I${BLAS_INCDIR}
ifeq "${CASPER}" "1"
CFLAGS += -DUSE_CSPR -DDEBUG -DCSPG_DEBUG    
INC += -I${CSPR_INCDIR}
endif
else
INC = -I./                  \
      -I../install          \
      -I${ERD_INCDIR}       \
      -I${MPI_INCDIR}       \
      -I${SCALAPACK_INCDIR} \
      -I${BLAS_INCDIR}
endif

PFOCK_LIBS = ../install/libpfock.a

ifeq "${ELEMENTAL}" "1"
CFLAGS += -DUSE_ELEMENTAL 
LDFLAGS = -L${MPI_LIBDIR}       \
          -L${SCALAPACK_LIBDIR} \
          -L${BLAS_LIBDIR}      \
          -L${EL_LIBDIR}        \
          -L${ERD_LIBDIR}
LIBS = ${MPI_LIBS}       \
       ${SCALAPACK_LIBS} \
       ${BLAS_LIBS}      \
       ${PFOCK_LIBS}     \
       ${ERD_LIBS}       \
       ${EL_LIBS}        \
       -lirc -lifcore -limf
ifeq "${CASPER}" "1"
LDFLAGS += -L${CSPR_LIBDIR}
LIBS += ${CSPR_LIBS}
endif
else
LDFLAGS = -L${MPI_LIBDIR}       \
          -L${GA_LIBDIR}        \
          -L${ARMCI_LIBDIR}     \
          -L${SCALAPACK_LIBDIR} \
          -L${BLAS_LIBDIR}      \
          -L${ERD_LIBDIR}
LIBS = ${MPI_LIBS}       \
       ${SCALAPACK_LIBS} \
       ${BLAS_LIBS}      \
       ${PFOCK_LIBS}     \
       ${ERD_LIBS}       \
       ${GA_LIBS}        \
       ${ARMCI_LIBS}     \
       -lirc -lifcore -limf
endif

all: ${EXEC} 

$(EXEC): $(OBJS) Makefile ${CINT_LIBS} ${PFOCK_LIBS}
	$(MPICC) ${CFLAGS} $(OBJS) -o $(EXEC) ${LDFLAGS} ${LIBS}

%.o : %.c Makefile
	$(MPICC) ${CFLAGS} ${INC} -c $< -o $@ 

clean:
	rm -f *.o *.s *.d *~ ${EXEC}
