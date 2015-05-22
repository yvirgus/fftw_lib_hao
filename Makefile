LIB     := libfftwhao.a

CC      := mpic++ -DMPI_HAO 
runtest := mpirun -np 4 runtest
#CC      := c++
#runtest := ./runtest



FLAGS   := -Wall -O3 -std=c++11 -I./inc
MPIFLG  := 
MPILIB  := 
FFTFLG  :=-I/opt/fftw/include
FFTLIB  :=-L/opt/fftw/lib -lfftw3 -lm

FLAGSALL:= $(FLAGS) $(MPIFLG) $(FFTFLG) 
LIBSALL :=          $(MPILIB) $(FFTLIB)

SRCS      := $(wildcard src/*.cpp)
SRCOBJS   := ${SRCS:.cpp=.o}
TESTS     := $(wildcard test/*.cpp)
TESTOBJS  := ${TESTS:.cpp=.o}

.PHONY: all clean distclean

all: test/runtest

test/runtest: lib/$(LIB) $(TESTOBJS)
	$(CC) $(FLAGSALL) -o runtest $(TESTOBJS) lib/$(LIB) $(LIBSALL)
	rm -rf test/runtest
	mv runtest test

lib/$(LIB): $(SRCOBJS)
	ar -rv $(LIB) $(SRCOBJS)
	rm -rf lib
	mkdir lib
	mv $(LIB) lib

check:
	(cd test;$(runtest));

rmdat:
	rm -f *.dat

clean:
	rm -rf lib
	rm -rf $(SRCOBJS)
	rm -rf test/runtest
	rm -rf $(TESTOBJS)

distclean: clean
	rm -f *.o  *.mod *.out *.dat *~


src/%.o :src/%.cpp
	$(CC) $(FLAGSALL) -c -o $@ $<

test/%.o :test/%.cpp
	$(CC) $(FLAGSALL) -c -o $@ $<


define OBJECT_DEPENDS_ON_CORRESPONDING_HEADER
    $(1) : ${1:src/%.o=inc/%.h}
endef

$(foreach object_file,$(SRCOBJS),$(eval $(call OBJECT_DEPENDS_ON_CORRESPONDING_HEADER,$(object_file)))) 
