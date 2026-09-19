CC      = mpicc
CFLAGS  = -O2 -Wall -Wextra
LDLIBS  = -lm

all: task1 task2

task1: task1.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

task2: task2.c
	$(CC) $(CFLAGS) -fopenmp -o $@ $< $(LDLIBS)

clean:
	rm -f task1 task2 primes.txt

.PHONY: all clean
