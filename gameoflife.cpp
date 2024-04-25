#include <iostream>
#include <random>
#include <chrono>
#include <ctime>

using std::cout; using std::endl;



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


void nextGeneration(bool* board, bool* next_board, int board_size){
}

int main()
{
    int board_size = 1024;
    int print_range = 64;
    bool *pre_board = new bool[board_size * board_size];
    bool *next_board = new bool[board_size * board_size];

    init_board(pre_board, board_size);

    // run at least ten generations and measure the time for each generation
    for (int i = 0; i < 10; i++){
        auto start = std::chrono::high_resolution_clock::now();
        
        // Run one generation on CPU
        nextGeneration(pre_board, next_board, board_size);
        
        // Swap the boards for the next iteration
        std::swap(pre_board, next_board);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        cout << "Time : Generation " << i << " took " << milliseconds.count() << " ms" << endl;
        
        if (i < 1) {
            print_board(pre_board, board_size, print_range);
        }
    }


    return 0;
}
