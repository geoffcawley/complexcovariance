#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <complex>
using namespace std;


#include <cuda_runtime.h>
//#include <helper_cuda.h>
#include <cuComplex.h>

/*
 * Do einsum('lmn,lmo->lno', x, x.conj()) / M  in series
 */
complex<float>* doProblemInSeries(complex<float>* input, int L, int M, int N) {
	complex<float>* matrix = new complex<float>[L*N*N];

	for (int z = 0;z < L;z++) {
		for (int y = 0;y < N;y++) {
			for (int x = 0;x < N;x++) {
				complex<float> t = (0.0f, 0.0f);
				for (int i = 0;i < M;i++) {
					t += input[z*M*N + i * N + x] * conj(input[z*M*N + y + i * N]);
				}
				t /= M;
				matrix[z*N*N + y * N + x] = t;
				cout << t;
			}
			cout << endl;
		}
		cout << endl;
	}
	return matrix;
}

__global__ void doProblemKernel(cuComplex *in, cuComplex *out, int L, int M, int N) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < L*N*N) {
		int z = index / (N*N);
		int y = (index - (z*N*N)) / N;
		int x = index - ((y*N) + (z*N*N));
		cuComplex t = make_cuComplex(0, 0);
		for (int i = 0;i < M;i++) {
			t = cuCaddf(t, cuCmulf(in[z*M*N + i * N + x], cuConjf(in[z*M*N + y + i * N])));
		}
		t = cuCdivf(t, make_cuComplex(M, 0));
		out[index] = t;
		//out[index] = make_cuComplex(index,x);
	}
}

void cudaInfo() {
	int dev_count;
	cudaGetDeviceCount(&dev_count);
	cudaDeviceProp dev_prop;
	for (int i = 0; i < dev_count; i++) {
		cudaGetDeviceProperties(&dev_prop, i);
		cout << "max threads per block : " << dev_prop.maxThreadsPerBlock << endl;
		cout << "max block x dim : " << dev_prop.maxThreadsDim[0] << endl;
		cout << "max block y dim : " << dev_prop.maxThreadsDim[1] << endl;
		cout << "max block z dim : " << dev_prop.maxThreadsDim[2] << endl;
		cout << "max grid x dim : " << dev_prop.maxGridSize[0] << endl;
		cout << "max grid y dim : " << dev_prop.maxGridSize[1] << endl;
		cout << "max grid z dim : " << dev_prop.maxGridSize[2] << endl;
		cout << "warp size : " << dev_prop.warpSize << endl;
	}

	cout << "Done\n";
}

void loadMatrix(string filename, complex<float>* matrix, int* L, int* M, int* N) {

}

int main()
{
	cudaInfo();
	int L, M, N;

	ifstream infile; string line;
	infile.open("dat\\matrix-med.txt");
	infile >> L; infile >> M; infile >> N;
	complex<float>* matrix = new complex<float>[L*M*N];

	cout << "L,M,N:" << L << "," << M << "," << N << ",\n"
		<< "Input size " << L*M*N << " Out Size " << L*N*N
		<< endl;

	for (int l = 0; l < L; l++) {
		for (int m = 0; m < M; m++) {
			for (int n = 0; n < N; n++) {
				//infile >> matrix[l][m][n];
				infile >> matrix[l*M*N + m * N + n];
			}
		}
	}

	infile.close();


	//for (int l = 0; l < L; l++) {
	//	for (int m = 0; m < M; m++) {
	//		for (int n = 0; n < N; n++) {
	//			//cout << matrix[l][m][n];
	//			cout << matrix[l*M*N + m * N + n];
	//		}
	//		cout << endl;
	//	}
	//	cout << "\n\n";
	//}

	complex<float>* seriesResult = doProblemInSeries(matrix, L, M, N);

	for (int l = 0; l < L; l++) {
		for (int m = 0; m < N; m++) {
			for (int n = 0; n < N; n++) {
				//cout << matrix[l][m][n];
				cout << seriesResult[l*N*N + m * N + n];
			}
			cout << endl;
		}
		cout << "\n\n";
	}

	ofstream outfile;
	outfile.open("dat\\seriesoutput.txt", 'w');
	for (int l = 0; l < L; l++) {
		for (int m = 0; m < N; m++) {
			for (int n = 0; n < N; n++) {
				//cout << matrix[l][m][n];
				outfile << seriesResult[l*N*N + m * N + n];
			}
			outfile << endl;
		}
		outfile << endl;
	}
	outfile.close();

	cudaError_t err = cudaSuccess;
	cuComplex* h_input = (cuComplex*)matrix;
	cuComplex* d_input = NULL;
	cudaMalloc((void**)&d_input, L*M*N * sizeof(cuComplex));
	cuComplex* d_output = NULL;
	cudaMalloc((void**)&d_output, L*N*N * sizeof(cuComplex));
	cudaMemcpy(d_input, h_input, L*M*N * sizeof(cuComplex), cudaMemcpyHostToDevice);
	//(cuComplex*)matrix;
	dim3 blockSize(1024, 1, 1);
	dim3 gridSize((L*N*N) / blockSize.x + 1, 1, 1);

	cout << "L,M,N:" << L << "," << M << "," << N << ",\n"
		<< "Input size " << L*M*N << " Out Size " << L*N*N
		<< endl;

	if (L*N*N > 1024) {
		doProblemKernel<<<gridSize, blockSize>>>(d_input, d_output, L, M, N);
	}
	else {
		doProblemKernel<<<1, L*N*N>>>(d_input, d_output, L, M, N);
	}
	err = cudaGetLastError();
	cuComplex* h_output = (cuComplex*)malloc(L*N*N * sizeof(cuComplex));
	cudaMemcpy(h_output, d_output, L*N*N * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	complex<float>* kerneloutput = (complex<float>*)h_output;


	for (int l = 0; l < L; l++) {
		for (int m = 0; m < N; m++) {
			for (int n = 0; n < N; n++) {
				//cout << matrix[l][m][n];
				cout << kerneloutput[l*N*N + m * N + n];
			}
			cout << endl;
		}
		cout << endl;
	}

	outfile.open("dat\\kernelsoutput.txt", 'w');
	for (int l = 0; l < L; l++) {
		for (int m = 0; m < N; m++) {
			for (int n = 0; n < N; n++) {
				//cout << matrix[l][m][n];
				outfile << kerneloutput[l*N*N + m * N + n];
			}
			outfile << endl;
		}
		outfile << endl;
	}
	outfile.close();
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
