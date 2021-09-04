#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <iostream>
#include <cmath>
#include <string>
#include <fstream>

const double LRate = 0.01;
const int FeatureCount = 30;
const int TrainCount = 455;
const int TestCount = 114;
const int MaxIteration = 1000;

// Device functions.
__device__ double Sigmoid(double z)
{
	double y = 0;

	y = 1 / (1 + exp(-z));

	return y;
}

__device__ double LossFunction(double expResult, double actResult)
{
	double result = 0;

	if (expResult == 0)
	{
		result = -log(1 - actResult);
	}
	else if (expResult == 1)
	{
		result = -log(actResult);
	}

	return result;
}

// one dimemsional grid and block.
__global__ void MatrixTranspose(double* i_matrix, double* o_matrix, int width, int height)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < width * height)
	{
		o_matrix[threadIdx.x * height + blockIdx.x] = i_matrix[tid];
	}
	// else do nothing.
}

__global__ void ForwardPropagation(int w, int h, double* i_x, double* i_y, double* i_w, double* o_y, double* oDiff_y ,double *bias, double* cost)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < h)
	{
		o_y[tid] = 0.0;

		for (int j = 0; j < w; j++)
		{
			o_y[tid] += i_w[j] * i_x[(tid * w) + j] + *bias;
		}
		// End of the loop.	

		o_y[tid] = Sigmoid(o_y[tid]);

		oDiff_y[tid] = o_y[tid] - i_y[tid];

		cost[tid] = LossFunction(i_y[tid], o_y[tid]);

		//__syncthreads();
	}
}

// w = trainCount, h = featureCount
__global__ void BackwardPropagation(int w, int h, double *transpose, double* diffY, double *weight, double* bias, double *sumdifferance,double learningRate)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	double derivativeWeight = 0;

	if (tid < h)
	{
		derivativeWeight = 0.0;
		sumdifferance[tid] = 0;

		for (int j = 0; j < w; j++)
		{
			derivativeWeight += diffY[j] * transpose[(tid * w) + j];

			sumdifferance[tid] = sumdifferance[tid] + diffY[j];
		}
		// end of the loop.

		// Update weight.
		weight[tid] = weight[tid] - learningRate * derivativeWeight;

		__syncthreads();

		if (tid == 0)
		{
			for (int k = 1; k < TrainCount; k++)
			{
				sumdifferance[0] = sumdifferance[0] + sumdifferance[k];
			}
			// There is no need for "else" statement.
		}

		*bias = *bias - learningRate * (sumdifferance[0] / TrainCount);
	}
}

__global__ void Test(int w, int h, double* x, double* y, double* weight, double *bias)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ double o_y[TestCount];
	__shared__ double accuracy[TestCount];

	if (tid < h)
	{
		o_y[tid] = 0.0;

		for (int j = 0; j < w; j++)
		{
			o_y[tid] += weight[j] * x[(tid * w) + j] + *bias;
		}
		// End of the loop.	

		o_y[tid] = Sigmoid(o_y[tid]);

		if (o_y[tid] < 0.5)
		{
			o_y[tid] = 0;
		}
		else
		{
			o_y[tid] = 1;
		}

		accuracy[tid] = accuracy[tid] + abs(y[tid] - o_y[tid]);

		__syncthreads();

		if (tid == 0)
		{
			for (int k = 1; k < h; k++)
			{
				accuracy[0] = accuracy[0] + accuracy[k];
			}

			accuracy[0] = 100 - (accuracy[0] / h) * 100;

			printf("Test accuracy: %lf\n", accuracy[0] );
		}
	}
	// else do nothing.
}


// Host Functions.
using namespace std;

void ReadDataFromFile(string fileName, double* data);
void InitialWeightAndBias(double* weight, int weightCount, double& bias);

int main(void)
{
	// Data file path.
	string trainXPath = "data/x_train.txt", trainYPath = "data/y_train.txt", testXPath = "data/x_test.txt", testYPath = "data/y_test.txt";

	// Train and test file.
	double *host_X_Train, *host_X_Test, *host_Y_Train, *host_Y_Test, *host_weight, *host_y, * host_diffY , host_bias = 0 , host_cost = 0;
	double *dev_X_Train, *dev_Trans_X_Train, *dev_X_Test,  *dev_Y_Train,  *dev_Y_Test,  *dev_weight, *dev_y, * dev_diffY, *dev_bias = 0, *dev_cost, *dev_sumdifferance;

	// Host.
	cudaMallocHost((void**)&host_X_Train, FeatureCount * TrainCount * sizeof(double) );
	cudaMallocHost((void**)&host_X_Test,  FeatureCount * TestCount * sizeof(double) );
	cudaMallocHost((void**)&host_Y_Train, TrainCount * sizeof(double) );
	cudaMallocHost((void**)&host_Y_Test,  TestCount * sizeof(double) );
	cudaMallocHost((void**)&host_y, TrainCount * sizeof(double));
	cudaMallocHost((void**)&host_diffY, TrainCount * sizeof(double));
	cudaMallocHost((void**)&host_weight,  FeatureCount * sizeof(double) );

	// Device.
	cudaMalloc((void**)&dev_X_Train, FeatureCount * TrainCount * sizeof(double));
	cudaMalloc((void**)&dev_Trans_X_Train, FeatureCount * TrainCount * sizeof(double));
	cudaMalloc((void**)&dev_X_Test,  FeatureCount * TestCount * sizeof(double));
	cudaMalloc((void**)&dev_Y_Train, TrainCount * sizeof(double));
	cudaMalloc((void**)&dev_Y_Test,  TestCount * sizeof(double));
	cudaMalloc((void**)&dev_weight,  FeatureCount * sizeof(double));
	cudaMalloc((void**)&dev_y, TrainCount * sizeof(double));
	cudaMalloc((void**)&dev_diffY, TrainCount * sizeof(double));
	cudaMalloc((void**)&dev_cost, TrainCount * sizeof(double));
	cudaMalloc((void**)&dev_sumdifferance, TrainCount * sizeof(double));
	cudaMalloc((void**)&dev_bias, sizeof(double));

	// Read data from file.
	ReadDataFromFile(trainXPath, host_X_Train);
	ReadDataFromFile(testXPath, host_X_Test);
	ReadDataFromFile(trainYPath, host_Y_Train);
	ReadDataFromFile(testYPath, host_Y_Test);

	// Initial weight and bias.
	InitialWeightAndBias(host_weight, FeatureCount, host_bias);

	cudaMemcpy(dev_X_Train, host_X_Train, FeatureCount * TrainCount * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_X_Test,  host_X_Test,  FeatureCount * TestCount * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Y_Train, host_Y_Train, TrainCount * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Y_Test,  host_Y_Test,  TestCount * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_weight,  host_weight,  FeatureCount * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(&dev_bias,  &host_bias, sizeof(double), cudaMemcpyHostToDevice);

	// GPU calculation.
	float elapsed;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Train.
	for (int i = 0; i < MaxIteration; i++)
	{
		// Z = X.W + b;
		ForwardPropagation << < 8, 64 >> > (FeatureCount, TrainCount, dev_X_Train, dev_Y_Train, dev_weight, dev_y, dev_diffY, dev_bias, dev_cost);

		// Transpose.
		MatrixTranspose << < TrainCount, FeatureCount >> > (dev_X_Train, dev_Trans_X_Train, FeatureCount, TrainCount);

		// w = trainCount, h = featureCount
		BackwardPropagation << < 1, FeatureCount >> > (TrainCount, FeatureCount, dev_Trans_X_Train, dev_diffY, dev_weight, dev_bias, dev_sumdifferance, 0.01);
	}

	// TestCount - 114
	//Test << < 4, 32 >> > (FeatureCount, TestCount, dev_X_Test, dev_Y_Test, double* o_y, dev_weight, *dev_bias, double* accuracy)
	Test << < 4, 32 >> > (FeatureCount, TestCount, dev_X_Test, dev_Y_Test, dev_weight, dev_bias);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);

	printf("The elapsed time in gpu was %.2f ms\n", elapsed);
	
	cudaMemcpy(&host_bias, &dev_bias[0], sizeof(double), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	printf("gpu cost : %lf\n", host_bias);


	cudaFree(dev_X_Train);
	cudaFree(dev_Trans_X_Train);
	cudaFree(dev_X_Test);
	cudaFree(dev_Y_Train);
	cudaFree(dev_Y_Test);
	cudaFree(dev_weight);
	cudaFree(dev_y);
	cudaFree(dev_diffY);
	cudaFree(dev_cost);
	cudaFree(dev_bias);

	cudaFreeHost(host_X_Train);
	cudaFreeHost(host_X_Test);
	cudaFreeHost(host_Y_Train);
	cudaFreeHost(host_Y_Test);
	cudaFreeHost(host_y);
	cudaFreeHost(host_diffY);
	cudaFreeHost(host_weight);

	system("pause");
	return 0;
}
void ReadDataFromFile(string fileName, double* data)
{
	ifstream infile;
	infile.open(fileName);

	if (infile.is_open() == false)
	{
		cout << fileName << " could not be opened" << endl;
		return;
	}

	long long index = 0;

	while (!infile.eof())
	{
		infile >> data[index];
		index++;
	}
	// End of the loop.
	infile.close();

	cout << fileName << " successfully read" << endl;
}
void InitialWeightAndBias(double* weight, int weightCount, double& bias)
{
	bias = 0;
	for (int i = 0; i < weightCount; i++)
	{
		weight[i] = 0.01;
	}
	// end of the for loop.
}