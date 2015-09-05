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

INC = -I./                  \
      -I../install          \
      -I${ERD_INCDIR}       \
      -I${MPI_INCDIR}       \
      -I${SCALAPACK_INCDIR} \
      -I${BLAS_INCDIR} 

PFOCK_LIBS = ../install/libpfock.a

ifeq "${ELEMENTAL}" "1"
CFLAGS += -DUSE_ELEMENTAL 
LDFLAGS = -L${MPI_LIBDIR}       \
          -L${EL_LIBDIR}        \
          -L${SCALAPACK_LIBDIR} \
          -L${BLAS_LIBDIR}      \
          -L${ERD_LIBDIR}
LIBS = ${MPI_LIBS}       \
       ${BLAS_LIBS}      \
       ${SCALAPACK_LIBS} \
       ${PFOCK_LIBS}     \
       ${ERD_LIBS}       \
       ${EL_LIBS}        \
       -lifcore -limf
else
LDFLAGS = -L${MPI_LIBDIR}       \
          -L${GA_LIBDIR}        \
          -L${SCALAPACK_LIBDIR} \
          -L${BLAS_LIBDIR}      \
          -L${ERD_LIBDIR}
LIBS = ${MPI_LIBS}       \
       ${BLAS_LIBS}      \
       ${SCALAPACK_LIBS} \
       ${PFOCK_LIBS}     \
       ${ERD_LIBS}       \
       ${GA_LIBS}        \
       -lifcore -limf
endif

all: ${EXEC} 

$(EXEC): $(OBJS) Makefile ${CINT_LIBS} ${PFOCK_LIBS}
	$(MPICC) ${CFLAGS} ${LDFLAGS} $(OBJS) -o $(EXEC) ${LIBS}

%.o : %.c Makefile
	$(MPICC) ${CFLAGS} ${INC} -c $< -o $@ 

clean:
	rm -f *.o *.s *.d *~ ${EXEC}
