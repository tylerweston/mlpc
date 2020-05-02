//
// Created by tyler on 2020-04-27.
//

#ifndef MLPC_MLP_H  // ensure singleton
#define MLPC_MLP_H

#define RANDSEED    // do we want a random seed or a consistent one?
// remove or comment out for set seed

#ifndef RANDSEED
int seed = 13777;
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#ifdef RANDSEED
#include <time.h>
#endif
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "mlp.h"

//

typedef struct
{
    int r, c;
    double **values;
} matrix;

typedef struct
{
    int num_inputs;         // number of input units
    int num_hidden;         // number of hidden layers
    int *hidden_units;      // number of hidden units/layer
    int num_outputs;        // number of output layers
    matrix **weights;        // weights, 1 per hidden layer + 1 per output layer
    matrix **bias;           // biases, 1 per hidden layer + 1 per output layer
    // todo: momentum!
    // double* velocities
    // todo:
    // each layer will have a function pointer to an associated activation function
    // for now, just assume relu
    // each activation function will take in a matrix and return a matrix!
    // todo:
    matrix (*activation)(matrix*);   // pointer to an activation function, per layer
    matrix (*activation_prime)(matrix*);   // pointer to a activation back prop
} neural_net;

typedef struct {
    int num_inputs;         // number of input units
    int num_hidden;         // number of hidden layers
    int num_outputs;        // number of output layers
    int* hidden_units;      // number of hidden units/layer
} args;

typedef struct {
    matrix* grad_hidden;
    matrix* grad_weights;
    matrix* grad_biases;
} grads;

typedef struct {
    matrix* data;       // examples x features
    matrix* targets;    // examples x 1
} batch;

int image_size = 28 * 28;   // this is the number of features we have

// defines

#define TRAIN_SIZE 50000
#define VALIDATION_SIZE 10000

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

#define err(s) {\
        perror((s));\
        exit(EXIT_FAILURE); }

#define DEBUG true
#define OUTPUTS 10

double e = 2.71828183;
#define pi = 3.14159265359;

// #define uniform_const = 0.3989422804;    // 1 / sqrt(2 * pi), for normal dist

// pointer to training labels
uint8_t *labels;
// pointer to image data
uint8_t *images;

double unif(double min, double max);
double rand_scaled_double();
double scaleval(uint8_t value);

// matrix library stuff
matrix* new_matrix(int rows, int cols);
matrix* rand_matrix(int rows, int cols);
matrix* copy_matrix(matrix *m);
matrix* concat_matrices(matrix **ms, int n, int direction);
int free_matrix(matrix *m);
int print_matrix(matrix *m);
int print_matrices(matrix **ms, int n);
double get_value(matrix *m, int row, int col);
double set_value(matrix *m, int row, int col, double value);
matrix* mmul(matrix *m1, matrix *m2);
matrix* madd(matrix *m1, matrix *m2);
matrix* transpose(matrix *m);
matrix* scalar_mult(matrix* m, double scalar);
matrix* stack(matrix *m, int num);
//matrix* one_hot(int size, int active);
void mshape(matrix* m);

double dot(matrix* m1, matrix* m2);
matrix* mrelu(matrix* m);
//matrix* mrelu_prime(matrix* m);
matrix* msum(matrix* m, int direction);
matrix *rand_int_matrix(int rows, int cols, int max_int);

// activation functions
double relu(double z);
double relu_prime(double z);

// nn stuff
neural_net* init_net(args *a);
matrix* forward(neural_net* nn, matrix* inputs);
// todo: this is not what backprop does
//grads* backprop(matrix* inputs, matrix* grad_hidden, neural_net* nn);

batch* get_batch(int size, int features);
matrix* affine_forward(matrix* inputs, matrix* weights, matrix* bias);
grads* affine_backwards(matrix* grad_y, matrix* hidden_in, matrix* weights);
matrix* relu_backwards(matrix* grad_h, matrix* inputs);



uint8_t* read_labels(void);
uint8_t* read_images(void);

args* make_args(int num_in, int num_hid, int num_out, int* hid_units);
void print_args(args* a);

#endif //MLPC_MLP_H
