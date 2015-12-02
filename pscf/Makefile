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

ifeq "${CHECK_FOR_NAN}" "1"
CFLAGS += -D__CHECK_FOR_NAN__
endif

ifeq "${ELEMENTAL}" "1"
INC = -I./                  \
      -I../install          \
      -I${EL_INCDIR}        \
      -I${ERD_INCDIR}       \
      -I${MPI_INCDIR}       \
      -I${SCALAPACK_INCDIR} \
      -I${BLAS_INCDIR}
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
          -L${MPI_LIBSUBDIR}    \
          -L${SCALAPACK_LIBDIR} \
          -L${EL_LIBDIR}        \
          -L${BLAS_LIBDIR}      \
          -L${ERD_LIBDIR}
LIBS = ${MPI_LIBS}       \
       ${BLAS_LIBS}      \
       ${SCALAPACK_LIBS} \
       ${PFOCK_LIBS}     \
       ${ERD_LIBS}       \
       ${EL_LIBS}        \
       -lirc -lifcore -limf
else
LDFLAGS = -L${MPI_LIBDIR}       \
          -L${GA_LIBDIR}        \
          -L${ARMCI_LIBDIR}     \
          -L${SCALAPACK_LIBDIR} \
          -L${BLAS_LIBDIR}      \
          -L${ERD_LIBDIR}
LIBS = ${MPI_LIBS}       \
       ${BLAS_LIBS}      \
       ${SCALAPACK_LIBS} \
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
