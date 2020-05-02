// neural network in C with a simple matrix implementation

#include "mlp.h"
/*
 todo:
  -
 - change how matrices store their entries so it is contiguous in memory instead of ragged like it is now
 - automate cleanup of matrices
 - make matrices that can hold different types? ie. generic matrices
      - NO! Later!
 - change matrices into tensors?
 - strassen for matrix multiplication
 - how to set our activation function within our NN ??
 - start with SGD then learn how to batch and train!
 - cross-entropy!
      - different loss functions!
 - spin matrix stuff off into a different file
      - split up different functions and data structs
 - error handling in a better way!
 - how to pack args
 - weight updates with momentum??
*/

/*
 * Neural Network notes:
 * Input layer: input_size
 * hidden layer 0: weight matrix input_size x hidden_units_1 & hidden_units_1 x 1 bias vector
 * so each hidden layer has a weight matrix that is size (layer-1 x layer) (hidden 0 layer-1 is input_size)
 * and a bias vector that is layer_size x 1
 * output layer: same as a hidden layer? should we just treat the same way?
 * each layer can have a normalization function associated with it, we can use function pointers
 * to deal with this!!  (what format will the function pointers be?)
 *

 forward pass
"""Runs the forward pass.
    Args:
        model: Dictionary of all the weights.
        x:     Input to the network.
    Returns:
        var:   Dictionary of all intermediate variables.
    """
    z1 = Affine(x, model['W1'], model['b1'])
    h1 = ReLU(z1)
    z2 = Affine(h1, model['W2'], model['b2'])
    h2 = ReLU(z2)
    y = Affine(h2, model['W3'], model['b3'])
    var = {
        'x': x,
        'z1': z1,
        'h1': h1,
        'z2': z2,
        'h2': h2,
        'y': y
    }
    return var

        """Runs the backward pass.
    Args:
        model:    Dictionary of all the weights.
        err:      Gradients to the output of the network.
        var:      Intermediate variables from the forward pass.
    """
    dE_dh2, dE_dW3, dE_db3 = AffineBackward(err, var['h2'], model['W3'])
    dE_dz2 = ReLUBackward(dE_dh2, var['z2'])
    dE_dh1, dE_dW2, dE_db2 = AffineBackward(dE_dz2, var['h1'], model['W2'])
    dE_dz1 = ReLUBackward(dE_dh1, var['z1'])
    _, dE_dW1, dE_db1 = AffineBackward(dE_dz1, var['x'], model['W1'])
    model['dE_dW1'] = dE_dW1
    model['dE_dW2'] = dE_dW2
    model['dE_dW3'] = dE_dW3
    model['dE_db1'] = dE_db1
    model['dE_db2'] = dE_db2
    model['dE_db3'] = dE_db3
    pass

       """Update NN weights.
    Args:
        model:    Dictionary of all the weights.
        eps:      Learning rate.
        momentum: Momentum.
    """
    to_update = ['W1', 'W2', 'W3',
                'b1', 'b2', 'b3']

    for u in to_update:
        vel = model['velocities'][u]
        vel = momentum * vel + (1 - momentum) * model["dE_d" + u]
        # update weights
        model[u] = model[u] - eps * vel
        # update velocity
        model['velocities'][u] = vel

    ###########################
 */

args* make_args(int num_in, int num_hid, int num_out, int* hid_units)
// initalize our arguments and return an args struct
{
    args* temp = malloc(sizeof(args));
    temp->num_inputs = num_in;
    temp->num_hidden = num_hid;
    temp->num_outputs = num_out;
    temp->hidden_units=malloc(sizeof(int*) * num_hid);
    memcpy(temp->hidden_units, hid_units, sizeof(int*) * num_hid);
    return temp;
}

void print_args(args* a)
// display an argument struct
{
    printf("Arguments:\n");
    printf("Number of inputs: %d\n", a->num_inputs);
    printf("Number of outputs: %d\n", a->num_outputs);
    printf("Number of hidden layers: %d\n", a->num_hidden);
    printf("Units per hidden layer:\n");
    for (int i = 0; i < a->num_hidden; i++)
    {
        printf("Layer %d: %d units\n", i, a->hidden_units[i]);
    }

}

int main(void)
{
    printf("Setting seed...");
	 // Set a random seed
#ifdef RANDSEED
     time_t t;
     srand((unsigned) time(&t));
#else
	// ... or run with set seed
    srand(seed);
#endif
    printf("Done.\n");

     // get images and labels
    printf("Reading labels and images...");
    labels = read_labels();
    images = read_images();
    printf("Done.\n");

    // generate arguments
    printf("Making arguments...");
    int* hids = malloc(sizeof(int*) * 2);
    hids[0] = 32;
    hids[1] = 16;
    args* a = make_args(28*28, 2, 10, hids);
    print_args(a);
    printf("Done.\n");

    // setup neural network
    // todo: this currently does not work!
//    neural_net* nn = init_net(a);

    // train neural network on training data
    // todo: train here

    // ...what about validation?!

    // now test it out on some unseen data!
    // todo: test here

    // testing some matrix stuff for now

//	 matrix *k = rand_int_matrix(3, 2, 10);
//	 matrix *j = msum(k, 0);
//	 matrix *m = scalar_mult(j, 0.5);
//
//	 print_matrix(k);
//	 print_matrix(j);
//	 print_matrix(m);
//    matrix *m = rand_int_matrix(2, 5, 10);
//    matrix *s = stack(m, 5);
//    print_matrix(m);
//    print_matrix(s);
//    mshape(s);
    batch* b = get_batch(5, image_size);
    printf("training data:\n");
    print_matrix(b->data);
    printf("targets:\n");
    print_matrix(b->targets);

    // clean up all alloc'd mem here
    free(labels);
    free(images);
	return 0;
}


// Matrix functions here
double dot(matrix* m1, matrix* m2)
// dot product between two vectors.
{
    // 1xn X nx1 -> scalar
    if (m1->r != 1 || m2->c != 1)
    {
        err("dimension mismatch, should be 1xn X nx1");
    }
    if (m1->c != m2->r)
    {
        err("dimension mismatch, should be 1xn X nx1");
    }
    int size = m1->c;
    double tot = 0.0;
    for (int i = 0 ; i < size; i++)
    {
        tot += m1->values[0][i] * m2->values[i][0];
    }
    return tot;
}

int free_matrix(matrix* m)
// free all pointers associated with a matrix
{
	for (int j = 0; j < m->r; j++)
	{
		free(m->values[j]);
	}
	free(m->values);
	free(m);
}

matrix *new_matrix(int rows, int cols)
// create a new matrix on the heap and return a pointer to it
// if runs into problems anywhere, throw an error!
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

matrix *rand_int_matrix(int rows, int cols, int max_int)
// return a rows x cols matrix with integer values between 0 - max_int
// since matrices only hold doubles now, these are cast to doubles, so
// not real ints!
{
    matrix *m = new_matrix(rows, cols);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            m->values[i][j] = (double) (rand() % max_int);
        }
    }
    return m;
}

matrix *rand_matrix(int rows, int cols)
// return a random real between 0.0 - 1.0
// should this be -1.0 to 1.0 by default??
// todo: maybe make the random number generator a function that we pass to this function?
// ie, we can pass any function that returns a double and we can use that to init our matrix
{
	matrix *m = new_matrix(rows, cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			m->values[i][j] = rand_scaled_double();
		}
	}
	return m;
}

void mshape(matrix* m)
{
    printf("Shape: %d x %d\n", m->r, m->c);
}

matrix* scalar_mult(matrix* m, double scalar)
// return a new matrix that is a scalar multiple of m
{
    matrix* temp = copy_matrix(m);
    for (int i = 0; i < m->r; i++)
    {
        for (int j = 0; j < m->c; j++)
        {
            temp->values[i][j] *= scalar;
        }
    }
    return temp;
}

matrix *copy_matrix(matrix *m)
// return a deep copy of a matrix
{
	matrix *temp = new_matrix(m->r, m->c);
    for (int i = 0; i < m->r; i++)
    {
        for (int j = 0; j < m->c; j++)
        {
            temp->values[i][j] = m->values[i][j];
        }
    }
	return temp;
}

matrix *concat_matrices(matrix **ms, int length_m, int direction)
// Concatenates matrices along a given direction
// todo: is there someway to do this in one pass? maybe dynamically hold and resize
// all entries as we check dimensions and then if everything is good just memcpy to the matrix struct?
{
	// direction = 0 concate downwards, so the cols of each matrix
	// must match, throw an error if they don't

	// direction = 1 concate sideways, so the rows of each matrix
	// must match
	// THIS IS NOT SUPPORTED YET
	if (direction == 1)
    {
	    err("Can only concatenate downwards for now!")
    }

	// grab target dimension from first matrix in out list of matrices

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

	//for (; mptr <  ms[0] + length_m; mptr++)  // hmm, this doesn't work, probably because matrices don't have
	                                            // a consistent size?
	for (int i = 0; i < length_m; i++)
	{
	    mptr = ms[i];
		if (DEBUG) print_matrix(mptr);
		curr_row = mptr->r;
		curr_col = mptr->c;

		if ((direction == 0 && target_cols != curr_col) ||
		    (direction == 1 && target_rows != curr_row))
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

    // todo: for now, only support concating down!
//	else if (direction == 1)
//	{
//		temp = new_matrix(target_rows, total_cols);
//	}
	// now, we fill in the values
	// might want to use offsetof to determine where each field is
	// in mem.
    int currow = 0;
    for (int i = 0; i < length_m; i++)
    {
        mptr = ms[i];
        for (int row = 0; row < mptr->r; row++)
        // todo: this is kind of gross for now, it would be better to do this as one
        // big memcpy but that will happen if these matrices are ever flattened down to
        // contiguous memory
        {
            memcpy(temp->values[currow], mptr->values[row], sizeof(double) * mptr->c);
            currow++;
        }
    }
    return temp;
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
		err("Dimension mismatch")
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
// display a given matrix
{
	printf("---\n");
//	printf("rows: %d cols: %d\n", m->r, m->c);
    int dispchl, dispchr;   // display character l, r
	for (int i = 0; i < m->r; i++)
	{
		// printf("| ");
		if (i == 0)
        {
            dispchl = '/';
            dispchr = '\\';
        }
		else if (i == m->r-1)
        {
            dispchl = '\\';
            dispchr = '/';
        }
		else
        {
            dispchl = '|';
            dispchr = '|';
        }
		printf("%c\t", dispchl);
		for (int j = 0; j < m->c; j++)
		{
			double value = get_value(m, i, j);
			printf("\t%lf ", value);
		}
		printf("\t%c\n", dispchr);
	}
}

double get_value(matrix *m, int row, int col)
// these may be useful again some day when I switch to contiguous mem
// for the matrices, but for now, they are unusued
{
	if (row < 0 || row > m->r ||
		col < 0 || col > m->c)
	{
		err("Index out of range");
	}
	return m->values[row][col];
}

double set_value(matrix *m, int row, int col, double value)
// these may be useful again some day when I switch to contiguous mem
// for the matrices, but for now, they are unusued
{
	if (row < 0 || row > m->r ||
		col < 0 || col > m->c)
	{
		err("Index out of range");
	}
	m->values[row][col] = value;
}

matrix* mrelu(matrix* m)
// element-wise rectified linear activation for matrix
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
	return return_matrix;
}

matrix* mrelu_prime(matrix* m)
// element-wise rectified linear activation derivative for matrix
{
    matrix *return_matrix = new_matrix(m->r, m->c);
    for (int row = 0; row<m->r; row++)
    {
        for (int col = 0; col<m->c; col++)
        {
            return_matrix->values[row][col] = relu_prime(m->values[row][col]);
        }
    }
    return return_matrix;
}

matrix* stack(matrix *m, int num)
// given an n x m matrix, return a (num*n) x m matrix which is just
// num copies of the original matrix on top of each other
// use this to broadcast the shape of the biases.
{
    matrix** temp = malloc(sizeof(matrix*) * num);
    for (int i = 0; i < num; i ++)
    {
        temp[i] = m;
    }
    return concat_matrices(temp, num, 0);
}

matrix* msum(matrix* m, int direction)
// sum down a matrix to a single dimension
// this can either be down in direction = 0 (downwards) or
// direction = 1 (across).
// direction = 0 means the resulting matrix will be 1 x total columns
// direction = 1 means the resulting matrix wil l be total rows x 1
{
    matrix* temp;
    double cursum;
    if (direction == 0)
    {
        temp = new_matrix(1, m->c);
        for (int i = 0; i < m->c; i++)
        {
            cursum = 0.0;
            for (int j = 0; j < m->r; j++)
            {
                cursum += m->values[j][i];
            }
            temp->values[0][i] = cursum;
        }
    }
    if (direction == 1)
    {
        temp = new_matrix(m->r, 1);
        // todo!
    }
    return temp;
}

// Math Stuff
double* get_softmax(matrix* v)
{
    // Return an array with the softmax values of the ouput
    double* soft_maxes = malloc(10 * sizeof(double));
    double total = 0.0;
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

double relu(double z)
// rectified linear activation function
{
    return max(0, z);
}

double relu_prime(double z)
// derivative of recitifed linear function
{
    return z > 0 ? 1 : 0;
}

double sigmoid(double z) 
{   // LOGISTIC
    // sigmoid function
    return (double) 1.0/(1.0 + pow(e, -z));
}

double sigmoid_prime(double z) 
{
    // derivative of sigmoid function put the fun in function!
    return (double) sigmoid(z) * (1 - sigmoid(z));
}

double rand_scaled_double() 
{
    // todo: this could look nicer?
    // return a double between -1.0 and 1.0
    return ((double)rand()/(double)(RAND_MAX/2)) - 1;
}

// uniform sampled
double unif(double min, double max)
{
    // return a random double between min and max
    // todo: test this still, code from the internet!
    return (max - min) * ( (double)rand() / (double)RAND_MAX ) + min;
}

double normal_dist(double mean, double sd)
{
    // todo: return a double sampled from a normal dist
    // icky, do this later
}

// init neural network
neural_net* init_net(args *a)
{
    neural_net *n = calloc(sizeof(neural_net), 1);
    n->hidden_units = calloc(sizeof(int), a->num_hidden);
    // todo: for now, initialize all hidden units with random values sampled from uniform dist

    // init weights and biases
    int rs, cs;
    for (int i = 0; i < a->num_hidden; i++)
    {
        if (i == 0)
        {
            // first layer, rows = size of input
            rs = a->num_inputs;
            cs = a->hidden_units[0];
        }
        else if (i == a->num_hidden)
        {
            // last layer, cols = size of output
            rs = a->hidden_units[1 - 1];
            cs = a->num_outputs;
        }
        else
        {
            // any other layer
            rs = a->hidden_units[i - 1];
            cs = a->hidden_units[i];
        }
        n->weights[i] = rand_matrix(rs, cs);
        n->bias[i] = rand_matrix(cs, 1);    // or init to zeros?
    }

//    // activation function stuff
//    n->activation = &mrelu; // set activation function
//    n->activation_prime = &mrelu_prime; // derivative of activation function

    return n;
}

matrix* forward(neural_net* nn, matrix* inputs)
{
    // todo: what is the size of the output of a forward pass?
    // it should be examples x output size!
    // todo: for now, just hard code shape of network,
    // eventually this can be made more arbitrary
    // to calculate first layer, mmult(weights, inputs) + msum(that + biases)
    matrix* z1 = affine_forward(inputs, nn->weights[0], nn->bias[0]);
    matrix* h1 = mrelu(z1);
    matrix* z2 = affine_forward(h1, nn->weights[1], nn->bias[1]);
    matrix* h2 = mrelu(z2);

}

matrix* affine_forward(matrix* inputs, matrix* weights, matrix* bias)
// affine (linear) forward pass for one layer
{
    matrix* temp = mmul(inputs, weights);                       // calculate wx
    matrix* stackbias = stack(transpose(bias), inputs->r);      // broadcast biases
    matrix* rval = madd(temp, stackbias);                       // wx + b
    free(temp);                                                 // cleanup
    free(stackbias);
    return rval;
}

grads* affine_backwards(matrix* grad_y, matrix* hidden_in, matrix* weights)
// returns gradients for affine layer
// returns a grad struct packed with matrices of:
//  gradient wrt inputs (grad_h)
//  gradient wrt weights (grad_w)
//  gradient wrt biases (grad_b)
{
    grads* temp = malloc(sizeof(grads));
    matrix* grad_h = mmul(grad_y, transpose(weights));          // grad_y.dot(w.T)
    matrix* grad_w = mmul(transpose(hidden_in), grad_y);        // h.T.dot(grad_y)
    matrix* grad_b = msum(grad_y, 0);                  // np.sum(grad_y, axis=0)
    temp->grad_hidden = grad_h;
    temp->grad_weights = grad_w;
    temp->grad_biases = grad_b;
    return temp;
}

matrix* relu_backwards(matrix* grad_h, matrix* inputs)
// grad of relu function. wrt inputs
{
    return mmul(grad_h, mrelu_prime(inputs));
}

grads* backprop(matrix* inputs, matrix* grad_hidden, neural_net* nn)
// back pass computes grad_h, grad_w, and grad_b
// pack this up in struct? or do in separate functions?
// grad_h = grad wrt. inputs/hidden layers
// grad_w = grads wrt. weights
// grad_b = grads wrt. biases
{

}

int train(neural_net* nn)
// main training loop for neural network
{
    int num_epochs = 100;   // this will go in args
    for (int epoch = 0; epoch < num_epochs; epoch++)
    {

    }
}




// MISC MATH.

double scaleval(uint8_t value)
{
    // returns a double mapped from 0-255 -> 0.0-1.0
    if (value == 0) return 0.0;
    return (double) (value / 255.0);
}

// DATA READING FUNCTIONS HERE:

batch* get_batch(int batch_size, int features)
// this will return a batch struct that includes
//  - an batch_size x features matrix loaded with our training data
//  - an batch_size x 1 matrix loaded with our targets
{
    batch* temp = malloc(sizeof(batch));
    matrix* train_data = new_matrix(batch_size, features);
    matrix* train_targets = new_matrix(batch_size, OUTPUTS);
    for (int curbatch = 0; curbatch < batch_size; curbatch++)
    {
        int tr = rand() % TRAIN_SIZE;                                   // choose random example
        train_targets->values[curbatch][labels[tr]] = 1.0;                      // get target
        int offs = tr * image_size;
        for (int im = 0; im < image_size; im++)
        {
            train_data->values[curbatch][im] = scaleval(images[offs + im]);
        }
    }
    temp->data = train_data;
    temp->targets = train_targets;
    return temp;
}

//matrix* one_hot(int size, int active)
//// return a 1 x size matrix with a 1 at position active, 0 everywhere else
//// This function is not used! (Or tested!!) (Yet?)
//{
//    matrix* temp = new_matrix(1, size);
//    temp->values[0][active] = 1.0;
//    return temp;
//}

uint8_t* read_images(void) 
{
    // after this all the labels have been read to
    // uint8 array images
    uint8_t *timages = NULL;
    timages = (uint8_t *) calloc(TRAIN_SIZE * image_size, sizeof(uint8_t));
    FILE *image_file = NULL;
    image_file = fopen("train_images.dat", "r");
    // skip to start of labels
    fseek(image_file, 16L, SEEK_SET);
    // read all data
    fread(timages, TRAIN_SIZE * image_size, 1, image_file);
    return timages;
}

uint8_t* read_labels(void) 
{
    // after this all the labels have been read to
    // uint8 array labels
    uint8_t *tlabels = NULL;
    tlabels = (uint8_t *) calloc(TRAIN_SIZE, sizeof(uint8_t));
    FILE *label_file = NULL;
    label_file = fopen("train_labels.dat", "r");
    // skip to start of labels
    fseek(label_file, 8L, SEEK_SET);
    // read all data
    fread(tlabels, TRAIN_SIZE, 1, label_file);
    return tlabels;
}
