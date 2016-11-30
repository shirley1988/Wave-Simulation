#include "mpi.h"     // mpi
#include <iostream>
#include "Timer.h"    
#include <stdlib.h>   // atoi
#include <math.h>     // pow
#include <stdio.h>
#include <omp.h>     // openmp
#include<pthread.h>  // pthread

int default_size = 100;  // the default system size
int defaultCellWidth = 8;
double c = 1.0;      // wave speed
double dt = 0.1;     // time quantum
double dd = 2.0;     // change in system

using namespace std;

// class ThreadParam: this class define the sole argument type for start_routing
// of pthread_create, it includes all parameters needed to invoke the 
// start_routine by the new thread.
class ThreadParam {
public:
    ThreadParam(double* z, int stripe, int size, int rank, int t, int n, 
    int thread_num): z(z), stripe(stripe), size(size), rank(rank), t(t), n(n),
    thread_num(thread_num) {};
    double* z;      // an array of double
    int stripe;     // stripe size
    int size;       // wave' one dimension size
    int rank;       // node's rank
    int t;          // time t
    int n;          // the amount of job needed to be done by slave threads
    int thread_num;  // number of slave threads
};

// calculate: method of calculating wave height at a specific time and point
void calculate(double* z, int size, int t, int k, int j ) {
    if ( k == 0 || j == 0 || j == size - 1 || k == size - 1 ) {      
                                         
       z[(t%3) * size * size + k * size + j] = 0;         // edge case
     } else {
       z[(t%3) * size * size + k * size + j] =            // internal case
          2.0 * z[((t - 1)%3) * size * size + k * size + j] - 
          z[((t - 2)%3) * size * size + k * size + j] +  c * c * pow(dt/dd, 2) * 
          (z[((t - 1)%3) * size * size + (k + 1) * size + j] + 
          z[((t - 1)%3) * size * size + (k - 1) * size + j] +
          z[((t - 1)%3) * size * size + k * size + j + 1] + 
          z[((t - 1)%3) * size * size + k * size + j - 1] -
          4.0 * z[((t - 1)%3) * size * size + k * size + j]);
     }
}

// calEge: this is the amount of job, calculating two edges' heights
// would be done by master thread, parameter n defines the work for master
// thread to do in only MPI case, that is to day, many nodes with 1 thread, 
// master do all the calculating job and n is equal to half of the stripe
void calEdge(double* z, int stripe, int size, int rank, int t, int n) {
   // should not be parallelized, master thread self sequentially do it
   for (int i  = 0; i < n; i++) {
       for (int j = 0; j < size; j++) { 
           int k = rank * stripe + i;          // the front edge
           int s = rank * stripe + stripe - 1 - i;  // the back edge
           calculate(z, size, t, k, j);
           calculate(z, size, t, s, j);
       }
   }
}

// calWave: calculate wave height from time = 2 to max_time.
// if thread_num is 0, then no need to implement openmp.
// whereas thread_num is larger than 0, split slave threads of number thread_num
// and calculate the wave height of a specific time. 
void calWave(double* z, int stripe, int size, int rank, int t, int n, 
 int thread_num) {
    if (thread_num == 0) {
        return;
    }
   omp_set_num_threads(thread_num);    // set thread_num of threads using openmp
   #pragma omp parallel for
   for (int i  = n; i < stripe - n; i++) {
       for (int j = 0; j < size; j++) { 
           int k = rank * stripe + i;      
           calculate(z, size, t, k, j);
       }
   }
}

// slaveJob: start_routine of pthread_create
void* slaveJob(void* arg) {
    ThreadParam &param = *(ThreadParam*)arg;
    calWave(param.z, param.stripe, param.size, param.rank, param.t, param.n, 
            param.thread_num);
}

// exchangeMessage: helper for simulateWaveWithInterval, 
// using MPI API to exchange mesages though internet
// only one node exist, no message exchanged
void exchangeMessage(double* z, int stripe, int size, int my_rank, int t) {
    int rank_num = size/stripe;
    if (rank_num == 1) {          // one node case
        return;
    }
     MPI_Status status;
    if (my_rank%2 == 0) {
      if ((my_rank - 1) >= 0) {
        MPI_Recv(z + ((t - 1)%3) * size * size + (my_rank * stripe - 1) * size,
                    size, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(z + ((t - 2)%3) * size * size + (my_rank * stripe - 1) * size,
                    size, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD, &status);
            
        MPI_Send(z + ((t - 1)%3) * size * size + (my_rank * stripe * size), 
                         size, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD);
        MPI_Send(z + ((t - 2)%3) * size * size + (my_rank * stripe * size),
                         size, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD);
      } 
      if ((my_rank + 1) < rank_num) {
        MPI_Recv(z + ((t - 1)%3) * size * size + ((my_rank + 1) * stripe)* size,
                     size, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(z + ((t - 2)%3)* size * size + ((my_rank + 1) * stripe) * size,
                     size, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD, &status);
            
        MPI_Send(z + ((t - 1)%3) * size * size + ((my_rank + 1) * stripe - 1) *
                       size, size, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD);
        MPI_Send(z + ((t - 2)%3) * size * size + ((my_rank + 1) * stripe - 1) * 
                        size, size, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD);
      }
    } else {
      if ((my_rank - 1) >= 0) {
        MPI_Send(z + ((t - 1)%3) * size * size + (my_rank * stripe * size), 
                         size, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD);
        MPI_Send(z + ((t - 2)%3) * size * size + (my_rank * stripe * size),
                         size, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD);
            
        MPI_Recv(z + ((t - 1)%3) * size * size + (my_rank * stripe - 1) * size,
                    size, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(z + ((t - 2)%3) * size * size + (my_rank * stripe - 1) * size,
                    size, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD, &status);
            
      }
      if ((my_rank + 1) < rank_num) {
        MPI_Send(z + ((t - 1)%3) * size * size + ((my_rank + 1) * stripe - 1) *
                       size, size, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD);
        MPI_Send(z + ((t - 2)%3) * size * size + ((my_rank  + 1) * stripe - 1) *
                       size, size, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD);
            
        MPI_Recv(z + ((t - 1)%3) * size * size + (my_rank + 1) * stripe * size,
                    size, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(z + ((t - 2)%3) * size * size + (my_rank + 1) * stripe * size, 
                    size, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD, &status);
      }
   }
}

// collect: collect data from slave processes and print out from master 
void collect(double* z, int stripe, int size, int my_rank, int t) {
    if (my_rank != 0) { // slave nodes sends data to master
        MPI_Send(z + (t%3) * size * size + my_rank * stripe * size, 
                 stripe * size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD );
    } else {
        MPI_Status status;  // master node receive data from slave nodes
        for (int i = 1; i < size/stripe; i++) {
            MPI_Recv(z + (t%3) * size * size + i * stripe * size, 
            stripe * size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
        }
    }
}

// print: print data though master process
void print(double* z, int stripe, int size, int my_rank, int t) {
    if (my_rank == 0) {
        cout << t << endl;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                cout << z[(t%3) * size * size + i * size + j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

// bestLoad: helper of deciding the workload of master thread
int bestLoad(int stripe, int thread_num) {
    if (thread_num == 1) {  // only master thread exist
        return stripe / 2 + 1;
    } else {         // slave threads exist
        return stripe / (thread_num * 4);
    }
}

// simulate wave diffusion from time = 2, and when interval is not 0, collect
// data if necessary and print from master nodes 
void simulateWaveWithInterval(double* z, int stripe, int size, int max_time, 
  int interval, int my_rank, int thread_num) {
    
    for (int t = 2; t < max_time; t++) {
        int n = bestLoad(stripe, thread_num);  // decide master thread's work load
        
        // the case when collecting data happens, when stripe == size, 
        // collect() returns
        if ( interval != 0 && (t % interval == 0 || t == max_time - 1) ) {
                collect(z, stripe, size, my_rank, t); 
                print(z, stripe, size, my_rank, t);
        }
        
        // create more slave threads by pthread_create and calculate internal cells
        pthread_t child1;
        ThreadParam* param = new ThreadParam(z, stripe, size, my_rank, t, n, thread_num - 1);
        pthread_create(&child1, NULL, slaveJob, (void*)param);
        //exchange edge data
        exchangeMessage(z, stripe, size, my_rank, t);
        // calculte edge data
        calEdge(z, stripe, size, my_rank, t, n);

        pthread_join(child1, NULL);
        
    }
}
  
int main( int argc, char *argv[] ) {
  // verify arguments
  if ( argc < 4 ) {
    cerr << "usage: Wave2D size max_time interval" << endl;
    return -1;
  }
  int size = atoi( argv[1] );
  int max_time = atoi( argv[2] );
  int interval  = atoi( argv[3] );
  int thread_num = 1;
  if (argc > 4) {
     thread_num = atoi(argv[4]);
  }

  if ( size < 100 || max_time < 3 || interval < 0 ) {
    cerr << "usage: Wave2D size max_time interval" << endl;
    cerr << "       where size >= 100 && time >= 3 && interval >= 0" << endl;
    return -1;
  }
 
  int my_rank = 0;
  int mpi_size = 1;
  
  MPI_Init( &argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  
  // create a simulation space
  double *z = new double[3 * size * size]; 
  for (int i = 0; i < 3 * size * size; i++) {
      z[i] = 0.0;
  }

  // start a timer
 Timer time;
 time.start( );

  // time = 0;
  // all mpi nodes compute time = 0; 
  int weight = size / default_size;
  for( int i = 0; i < size; i++ ) {
    for( int j = 0; j < size; j++ ) {
      if( i > 40 * weight && i < 60 * weight && j > 40 * weight && j < 60 * weight ) {
	     z[i * size + j] = 20.0;
      } else {
	     z[i * size + j] = 0.0;
      }
    }
  }
  
  int stripe = size/mpi_size;
  
  // time = 1, calculate z[1][][] 
  // all mpi nodes compute time = 1;
  for (int i  = 0; i < size; i++) {
       for (int j = 0; j < size; j++) {           
           if (i == 0 || j == 0 || j == size - 1 || i == size - 1) {      
                                         
               z[(size + i) * size + j] = 0;
           } else {
               z[(size + i) * size + j] = z[i * size + j] + 
                   c * c/2 * pow(dt/dd, 2) * ( z[(i + 1) * size + j] + 
                 z[(i - 1) * size + j] + z[i * size + j + 1]
            + z[i * size + j - 1] - 4.0 * z[i * size + j]);
           }
       }
   }
 
 // calculate time = 2 to max_time
simulateWaveWithInterval(z, stripe, size, max_time, interval, my_rank, thread_num);

 // deallocate memory
 delete[] z;

 // finish the timer

 if (my_rank == 0) {
      cerr << "my rank: " << my_rank << " elapsed time = " << time.lap( ) << endl;
 }
 
 MPI_Finalize();
 return 0;
}

