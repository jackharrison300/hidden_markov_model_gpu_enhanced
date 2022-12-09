#include <math.h>

__device__ float distance(float x1, float y1, float x2, float y2) {
    return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

// taken from https://stackoverflow.com/questions/10847007/using-the-gaussian-probability-density-function-in-c
__device__ float normal_pdf(float x, float m, float s) {
    static const float inv_sqrt_2pi = 0.3989422804014327;
    float a = (x - m) / s;
    return inv_sqrt_2pi / s * exp(-0.5f * a * a);
}

__global__ void observe_ei_gpu(float *landmarks, float *observed_dists, float *belief,
    int rows, int cols, float noise) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (r < rows && c < cols) {
        const float x_coord = (c+0.5) / (float) cols;
        const float y_coord =  (rows-1-r+0.5) / (float) rows;

        float prob = 1.0;
        // iterate through landmarks
        for (int i=0; i<4; i++) {
            const float mean = distance(x_coord, y_coord, landmarks[i*2], landmarks[i*2+1]);
            prob *= normal_pdf(observed_dists[i], mean, noise);
        }
        belief[r*cols+c] *= prob;
    }
}

// cuda recreation of Inference.transition_model
// updates transitions and returns the number of transitions
__device__ int transition_model(int r, int c, int rows, int cols, int* transitions, float* probs) {
    int num_trans = 0;
    if (r > 0) {
        transitions[0] = r - 1;
        transitions[1] = c;
        num_trans++;
    }
    if (r < rows - 1) {
        transitions[num_trans*2] = r + 1;
        transitions[num_trans++*2+1] = c;
    }
    if (c > 0) {
        transitions[num_trans*2] = r;
        transitions[num_trans++*2+1] = c - 1;
    }
    if (c < cols - 1) {
        transitions[num_trans*2] = r;
        transitions[num_trans++*2+1] = c + 1;
    }
    if (r > 0 && c > 0) {
        transitions[num_trans*2] = r - 1;
        transitions[num_trans++*2+1] = c - 1;
    }
    if (r < rows - 1 && c > 0) {
        transitions[num_trans*2] = r + 1;
        transitions[num_trans++*2+1] = c - 1;
    }
    if (r > 0 && c < cols - 1) {
        transitions[num_trans*2] = r - 1;
        transitions[num_trans++*2+1] = c + 1;
    }
    if (r < rows - 1 && c < cols - 1) {
        transitions[num_trans*2] = r + 1;
        transitions[num_trans++*2+1] = c + 1;
    }
    transitions[num_trans*2] = r;
    transitions[num_trans++*2+1] = c;

    const float prob = 1.0 / (float) num_trans;
    // note that probability per trans is constant, so for this particular transition model
    // we could just set one variable with probability. However, setting probability per trans
    // enables us to add complexity to the model later
    for (int i=0; i<num_trans; i++) {
        probs[i] = prob;
    }

    return num_trans;
}

__global__ void timeupdate_ei_gpu(float *belief, float *next_belief, int rows, int cols) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows && c < cols) {
        // transitions, mimicking 2D array
        // outer array has size 9 - max possible num transitions (one in each direction + stay in place)
        // inner array has size 2 - r, c
        int transitions[18];
        // probabilities, one per trans
        float probs[9];
        int num_trans = transition_model(r, c, rows, cols, transitions, probs);

        for (int i=0; i<num_trans; i++) {
            const int r_next = transitions[i*2];
            const int c_next = transitions[i*2+1];
            const float prob = probs[i];
            printf("%f ", prob);
            // add p(x_{t+1}|x_t)p(x_t|e_{1:t}) to current state of p(x_{t+1}|e_{1:t})
            atomicAdd(&next_belief[r_next*cols+c_next], prob * belief[r*cols+c]);
        }
    }
}