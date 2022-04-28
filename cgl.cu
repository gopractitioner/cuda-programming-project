#include <iostream>
#include <random>
#include <chrono>
#include <sys/time.h>
#include <ctime>


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
     
  for (int row = 1; row < board_size -1; row++) {
     for(int col= 1; col < board_size -1; col++){           
         
        }
     }

}
/**
* Implemention of the GPU version without using shared memory
*
*/
__global__ void nextGenerationGPU(bool* board, bool* next_board, int board_size){

}

/**
* Implemention of the GPU version using shared memory   
*
*/
__global__ void nextGenerationGPUSharedMemory(bool* board, bool* next_board, int board_size){
     
}

int main(void)
{
  int board_size = 1024; //2048,4096,8192,16384,32768
  int print_range = 64;
  bool *pre_board; 
  bool *next_board;

   

  for (int i=0; i<10;i++){

      //run at least ten generations and measure the elapsed time for each generation
      auto start = std::chrono::high_resolution_clock::now();
       
      //run one generation

      auto end = std::chrono::high_resolution_clock::now();
      auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
 
     }

  return 0;
}

