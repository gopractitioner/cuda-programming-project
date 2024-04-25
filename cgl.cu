#include <iostream>
#include <random>
#include <chrono>
#include <sys/time.h>
#include <ctime>
#include <cuda_runtime.h>


using std::cout; using std::endl;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;


/**
* range: the maximum of rows and columns of the board to print 
*/
void print_board(bool *board, int board_size, int range){
   cout << endl;
   for (int row = 0; row < range && row < board_size; row++) {
     for(int col=0; col < range && col < board_size; col++){
           cout << board[col + row*board_size];

     }
     cout<<endl;
   }
    
}

/**
* initialize a board array of board_size*board_size on the host
*
*/
void init_board(bool* board, int board_size){
  //starts with a simple pattern at the left corner region (0,0) to (range, range)
  int range = 128;
  //use a fixed seed for the same board pattern 
  srand(1);
  //get system time in seconds and use it as the random seed for different board patterns       
  //auto sec_since_epoch = duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
  //srand(sec_since_epoch);
  for (int row = 1; row < board_size -1; row++) {
     for(int col= 1; col < board_size -1; col++){
         if(row < range && col < range){
               board[col + row * board_size] = rand()%2;
               continue;                      
        }
       board[col + row * board_size] = 0;
     }
  }
}

/**
*Implemention of the CPU version
*Any live cell with fewer than two live neighbours dies, as if by underpopulation.
*Any live cell with two or three live neighbours lives on to the next generation.
*Any live cell with more than three live neighbours dies, as if by overpopulation.
*Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
*Note: you can ignore the cells on four edges by setting row or col = 1 and row or col < board_size-1
*/
void nextGeneration(bool* board, bool* next_board, int board_size){
     
  // for (int row = 1; row < board_size -1; row++) {
  //    for(int col= 1; col < board_size -1; col++){           
         
  //       }
  //    }
  int dx[8] = {0, 0, 1, -1, 1, 1, -1, -1};
  int dy[8] = {1, -1, 0, 0, 1, -1, 1, -1};

  for (int row = 1; row < board_size - 1; row++) {
      for (int col = 1; col < board_size - 1; col++) {
          int live_neighbors = 0;
          for (int i = 0; i < 8; i++) {
              int new_row = row + dy[i];
              int new_col = col + dx[i];
              live_neighbors += board[new_col + new_row * board_size];
          }
          bool current_cell = board[col + row * board_size];
          if (current_cell && (live_neighbors < 2 || live_neighbors > 3))
              next_board[col + row * board_size] = false;
          else if (!current_cell && live_neighbors == 3)
              next_board[col + row * board_size] = true;
          else
              next_board[col + row * board_size] = current_cell;
      }
  }

}
/**
* Implemention of the GPU version without using shared memory
*
*/
__global__ void nextGenerationGPU(bool* board, bool* next_board, int board_size){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= 1 && row < board_size - 1 && col >= 1 && col < board_size - 1) {
      int live_neighbors = 0;

      // Check all eight neighbors
      for (int y = -1; y <= 1; y++) {
          for (int x = -1; x <= 1; x++) {
              if (x == 0 && y == 0) continue;
              live_neighbors += board[(row + y) * board_size + (col + x)];
          }
      }

      bool current_cell = board[row * board_size + col];
      bool next_cell = current_cell;

      if (current_cell && (live_neighbors < 2 || live_neighbors > 3))
          next_cell = false; // Cell dies
      else if (!current_cell && live_neighbors == 3)
          next_cell = true; // Cell becomes alive

      next_board[row * board_size + col] = next_cell;
  }
  
}

/**
* Implemention of the GPU version using shared memory   
*Running a CUDA project in VSCode requires some initial setup, especially around build and debug configurations, but provides a powerful environment for developing GPU-accelerated applications.
*/
__global__ void nextGenerationGPUSharedMemory(bool* board, bool* next_board, int board_size){
     
}

int main() {

    int board_size = 1024; // Adjust size as needed
    int print_range = 64;
    size_t bytes = board_size * board_size * sizeof(bool);

    // Host allocation
    bool *pre_board = new bool[board_size * board_size];
    bool *next_board = new bool[board_size * board_size];

    // Device allocation
    bool *d_pre_board, *d_next_board;
    cudaMalloc(&d_pre_board, bytes);
    cudaMalloc(&d_next_board, bytes);

    // Initialize boards on host or use any other init function
    // Assuming init_board() fills `pre_board`
    init_board(pre_board, board_size);

    // Copy data from host to device
    cudaMemcpy(d_pre_board, pre_board, bytes, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 block(16, 16);  // Adjust block size as needed
    dim3 grid((board_size + block.x - 1) / block.x, (board_size + block.y - 1) / block.y);

    // // Run kernel and measure time
    // auto start = std::chrono::high_resolution_clock::now();

    // nextGenerationGPU<<<grid, block>>>(d_pre_board, d_next_board, board_size);

    // cudaDeviceSynchronize(); // Wait for GPU to finish

    // auto end = std::chrono::high_resolution_clock::now();
    // auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // std::cout << "time: " << milliseconds.count() << " ms" << std::endl;
    // print_board(pre_board, board_size, print_range);



    for (int i = 0; i < 10; i++) {
      auto start = std::chrono::high_resolution_clock::now();
      nextGenerationGPU<<<grid, block>>>(d_pre_board, d_next_board, board_size);
      cudaDeviceSynchronize();
      auto end = std::chrono::high_resolution_clock::now();

      // Copy results back to host 
      cudaMemcpy(pre_board, d_next_board, bytes, cudaMemcpyDeviceToHost);

      std::cout << "\nGeneration " << i + 1 << endl;

      auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      std::cout << "Time length: " << microseconds.count() << " microseconds";

      print_board(pre_board, board_size, print_range);

      std::swap(d_pre_board, d_next_board);

  }




    // Copy results back to host
    //cudaMemcpy(next_board, d_next_board, bytes, cudaMemcpyDeviceToHost);



    // clean memory
    cudaFree(d_pre_board);
    cudaFree(d_next_board);

    delete[] pre_board;
    delete[] next_board;

    return 0;
}

