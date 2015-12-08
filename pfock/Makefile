include ../make.in

LIBPFOCK = libpfock.a

SRC = $(wildcard *.c)
OBJS := $(addsuffix .o, $(basename $(SRC)))

ifeq "${DYNAMIC}" "1"
CFLAGS += -D__DYNAMIC__
endif

ifeq "${DO_SCF}" "1"
CFLAGS += -D__SCF__
endif

ifeq "${ELEMENTAL}" "1"
CFLAGS += -DUSE_ELEMENTAL
INC = -I.                   \
      -I${MPI_INCDIR}       \
      -I${SCALAPACK_INCDIR} \
      -I${BLAS_INCDIR}      \
      -I${EL_INCDIR}        \
      -I${ERD_INCDIR}
ifeq "${CASPER}" "1"
CFLAGS += -DUSE_CASPER -DDEBUG -DCSPG_DEBUG    
INC += -I${CSPR_INCDIR}
endif
else
INC = -I.                   \
      -I${MPI_INCDIR}       \
      -I${SCALAPACK_INCDIR} \
      -I${BLAS_INCDIR}      \
      -I${GA_INCDIR}        \
      -I${ERD_INCDIR}
endif

ifeq "${ELEMENTAL}" "1"
all: ${LIBPFOCK}
	mkdir -p ../install
	cp -r ${LIBPFOCK} ../install/
	cp -r pfock.h ../install/
	cp -r elem.h ../install/
else
all: ${LIBPFOCK}
	mkdir -p ../install
	cp -r ${LIBPFOCK} ../install/
	cp -r pfock.h ../install/
endif

${LIBPFOCK}: ${OBJS}
	${AR} $@ $^

%.o : %.c Makefile
	$(MPICC) ${CFLAGS} ${INC} -c $< -o $@

clean:
	rm -f *.o *.s *.d *~ *.a
