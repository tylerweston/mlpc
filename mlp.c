// neural network in C with a simple matrix implementation

// todo:
// - write matrix copy function
// - automate cleanup of matrices
// - make matrices that can hold different types? ie. generic matrices
// - change matrices into tensors?
// - stack/concat function for matrices

// Keep list of all created matrices and then free them by one call?
// ie free_matrices

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

// 

typedef struct
{
	int r, c;
	double **values;

} matrix;



typedef matrix vector;

int image_size = 28 * 28;
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

double e = 2.71828183;

// pointer to training labels
uint8_t *labels;
// pointer to image data
uint8_t *images;


matrix *new_matrix(int rows, int cols);
matrix *rand_matrix(int rows, int cols);
matrix *copy_matrix(matrix *m);
matrix *concat_matrices(matrix **ms, int n, int direction);
int free_matrix(matrix *m);
int print_matrix(matrix *m);
int print_matrices(matrix **ms, int n);
double get_value(matrix *m, int row, int col);
double set_value(matrix *m, int row, int col, double value);
matrix *mmul(matrix *m1, matrix *m2);
matrix *madd(matrix *m1, matrix *m2);
matrix *transpose(matrix *m);
double rand_scaled_double();
double relu(double z);

uint8_t* read_labels(void);
uint8_t* read_images(void);


int main(void)
{
	// // Set a random seed
 //    time_t t;
 //    srand((unsigned) time(&t));

	// ... or run with set seed
    unsigned int seed = 13000;
    srand(seed);

    // // get images and labels
    // labels = read_labels();
    // images = read_images();

    // testing some matrix stuff for now
	// matrix *m = rand_matrix(2, 3);
	// matrix *mt = transpose(m);
	// matrix *mcopy = copy_matrix(m);
	// matrix *k = rand_matrix(3, 2);
	// matrix *l = rand_matrix(3, 2);
	matrix *cc1 = rand_matrix(2, 4);
	matrix *cc2 = rand_matrix(3, 4);
	matrix **matrices = malloc(sizeof(matrix*) * 2);
	matrices[0] = cc1;
	matrices[1] = cc2;

	matrix* concated = concat_matrices(matrices, 2, 0);
	print_matrix(concated);


	// matrix *j = mmul(m, k);
	// matrix *r = madd(k, l);

	// printf("Original:\n");
	// print_matrix(m);
	// printf("Transposed:\n");
	// print_matrix(mt);	
	// printf("Copy:\n");
	// print_matrix(mcopy);
	// print_matrix(k);
	// print_matrix(l);
	// printf("---\n");
	// print_matrix(j);
	// print_matrix(r);

	// free_matrix(m);
	// free_matrix(mt);
	// free_matrix(mcopy);
	// free_matrix(k);
	// free_matrix(j);
	// free_matrix(r);

	return 0;
}

inline double scale255(uint8_t value)
{
	// returns a double mapped from 0-255 -> 0.0-1.0
	return (double) 255 / value;
}

// Matrix functions here
int free_matrix(matrix* m)
{
	for (int j = 0; j < m->r; j++)
	{
		free(m->values[j]);
	}
	free(m->values);
	free(m);
}

matrix *new_matrix(int rows, int cols)
{
	// error check this
	matrix *m = malloc(sizeof(matrix));
	if (m == NULL)
	{
		err("Couldn't make matrix");
	}

	m->values = calloc(rows, sizeof(double *));
	if (m->values == NULL)
	{
		free(m);
		err("Couldn't make matrix");
	}

	for (int i = 0; i < rows; i++)
	{
		m->values[i] = calloc(cols, sizeof(double));
		if (m->values[i] == NULL)
		{
			for (int j = 0; j < i; j++)
			{
				free(m->values[j]);
			}
			free(m->values);
			free(m);
			err("Couldn't make matrix");
		}
	}

	m->r = rows;
	m->c = cols;

	return m;
}

matrix *rand_matrix(int rows, int cols)
// return a random real between 0.0 - 1.0
{
	matrix *m = new_matrix(rows, cols);
	m->r = rows; 
	m->c = cols;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			m->values[i][j] = rand_scaled_double();
		}
	}
	return m;
}

matrix *copy_matrix(matrix *m)
// return a copy of a matrix
{
	matrix *temp = new_matrix(m->r, m->c);
	memcpy(temp->values, m->values, sizeof(double) * m->r * m->c);
	return temp;
}

matrix *concat_matrices(matrix **ms, int length_m, int direction)
// Concatenates matrices along a given direction
{
	// direction = 0 concate downwards, so the cols of each matrix
	// must match, throw an error if they don't

	// direction = 1 concate sideways, so the rows of each matrix
	// must match

	// grab target dimension from first matrix in out list of matrices
	// print_matrix(ms[1]);
	matrix *mptr = ms[0];
	int target_rows = mptr->r;
	int target_cols = mptr->c;
	if (DEBUG)
	{
		printf("target_rows: %d target_cols: %d\n", target_rows, target_cols);
	}

	// now iterate over all matrices, checking and collecting dimensions
	int total_rows = 0, total_cols = 0;
	int curr_row = 0, curr_col = 0;

	for (; mptr <  ms[0] + length_m; mptr++)
	{
		print_matrix(mptr);
		curr_row = mptr->r;
		curr_col = mptr->c;
		if (DEBUG)
		{
			printf("curr_row: %d curr_col: %d\n", curr_row, curr_col);
		}

		if ((direction == 0) && (target_rows != curr_row) ||
			(direction == 1) && (target_cols != curr_col))
		{
			// Todo: More error logging here, specify which matrix
			// and dimensions are wrongs
			err("Dimension mismatch");
		}

		total_rows += curr_row;
		total_cols += curr_col;
	}
	if (DEBUG)
	{
		printf("Target rows: %d target cols: %d\n", target_rows, target_cols);
		printf("Total rows: %d total cols: %d\n", total_rows, total_cols);
	}
	// ok, so all dimensions match up, let's make our new matrix
	// of the correct size
	matrix *temp;
	if (direction == 0)
	{
		temp = new_matrix(total_rows, target_cols);
	}	
	else if (direction == 1)
	{
		temp = new_matrix(target_rows, total_cols);
	}
	// now, we fill in the values


}

matrix *mmul(matrix *m1, matrix *m2)
// multiplies two matrices together and returns a pointer
// to a new matrix on the heap
{
	if (m1->c != m2->r)
	{
		if (DEBUG)
		{
			printf("Mismatch: %d != %d\n",
				m1->c, m2->r);
		}
		// TODO: End program here? OR allow try to recover?
		err("Dimension mismatch");
		return NULL;
	}
	int rows = m1->r;
	int cols = m2->c;
	matrix *m = new_matrix(rows, cols);
	m->r = rows;
	m->c = cols;
	double sum;	
	for (int i=0; i<rows; i++) 
	{
		for (int j=0; j<cols; j++)
		{
			sum = 0;
			for (int k=0; k<m2->r; k++)
			{
				sum += m1->values[i][k] * m2->values[k][j];
			}
			m->values[i][j] = sum;
		}
	}
	return m;

}

matrix *madd(matrix *m1, matrix *m2)
{
	if (m1->r != m2->r ||
		m1->c != m2->c)
	{
		err("Dimension mismatch")
		return NULL;
	}
	int rows = m1->r;
	int cols = m1->c;
	matrix *m = new_matrix(rows, cols);
	m->r = rows;
	m->c = cols;
	for (int i = 0; i < m->r; i++)
	{
		for (int j = 0; j < m->c; j++)
		{
			double value = get_value(m1, i, j) + get_value(m2, i, j);
			set_value(m, i, j, value);
		}
		printf("\n");
	}
	return m;
}

matrix *transpose(matrix *m)
{
	matrix *rm = new_matrix(m->c, m->r);
	for (int i = 0; i<m->r; i++)
	{
		for (int j = 0; j<m->c; j++)
		{
			rm->values[j][i] = m->values[i][j];
		}
	}
	return rm;
}

int print_matrix(matrix *m)
{
	printf("---\n");
	printf("rows: %d cols: %d\n", m->r, m->c);
	for (int i = 0; i < m->r; i++)
	{
		printf("| ");
		for (int j = 0; j < m->c; j++)
		{
			double value = get_value(m, i, j);
			printf("\t%lf ", value);
		}
		printf("\t|\n");
	}
}

double get_value(matrix *m, int row, int col)
{
	if (row < 0 || row > m->r ||
		col < 0 || col > m->c)
	{
		err("Index out of range");
	}
	return m->values[row][col];
}

double set_value(matrix *m, int row, int col, double value)
{
	if (row < 0 || row > m->r ||
		col < 0 || col > m->c)
	{
		err("Index out of range");
	}
	m->values[row][col] = value;
}

double mrelu(matrix* m)
{
	matrix *return_matrix = new_matrix(m->r, m->c);
	return_matrix->r = m->r;
	return_matrix->c = m->c;
	for (int row = 0; row<m->r; row++)
	{
		for (int col = 0; col<m->c; col++)
		{
			return_matrix->values[row][col] = relu(m->values[row][col]);
		}
	}
}

// Math Stuff
double* get_softmax(vector* v)
{
    // Return an array with the softmax values of the ouput
    double* soft_maxes = malloc(10 * sizeof(double));
    double total;
    for (int i = 0; i < v->r; i++)
    {
        total += pow(e, v->values[i][0]);
    }
    for (int i = 0; i < v->r; i++)
    {
        soft_maxes[i] = pow(e, v->values[i][0]) / total;
    }
    if (DEBUG)
    {
        for (int i = 0; i < v->r; i++)
        {
            printf("Probability of %d is %f\n", i, soft_maxes[i]);
        }
    }
    return soft_maxes;
}

double relu(double z)   // rectified linear
{
    return max(0, z);
}

double relu_prime(double z)
{
    return z > 0 ? z : 0;
}

double sigmoid(double z) 
{   // LOGISTIC
    // sigmoid function
    return (double) 1.0/(1.0 + pow(e, -z));
}

double sigmoid_prime(double z) 
{
    // derivative of sigmoid function put the fun in funccton!
    return (double) sigmoid(z) * (1 - sigmoid(z));
}

double rand_scaled_double() 
{
    // todo: this could look nicer?
    // return a double between -1.0 and 1.0
    return ((double)rand()/(double)(RAND_MAX/2)) - 1;
}

// DATA READING FUNCTIONS HERE:

uint8_t* read_images(void) 
{
    // after this all the labels have been read to
    // uint8 array images
    uint8_t *images = NULL;
    images = (uint8_t *) calloc(TRAIN_SIZE * image_size, sizeof(uint8_t));
    FILE *image_file = NULL;
    image_file = fopen("train_images.dat", "r");
    // skip to start of labels
    fseek(image_file, 16L, SEEK_SET);
    // read all data
    fread(images, TRAIN_SIZE * image_size, 1, image_file);
    return images;
}

uint8_t* read_labels(void) 
{
    // after this all the labels have been read to
    // uint8 array labels
    uint8_t *labels = NULL;
    labels = (uint8_t *) calloc(TRAIN_SIZE, sizeof(uint8_t));
    FILE *label_file = NULL;
    label_file = fopen("train_labels.dat", "r");
    // skip to start of labels
    fseek(label_file, 8L, SEEK_SET);
    // read all data
    fread(labels, TRAIN_SIZE, 1, label_file);
    return labels;
}
