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

/**
*Implemention of the CPU version
*Any live cell with fewer than two live neighbours dies, as if by underpopulation.
*Any live cell with two or three live neighbours lives on to the next generation.
*Any live cell with more than three live neighbours dies, as if by overpopulation.
*Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
*Note: you can ignore the cells on four edges by setting row or col = 1 and row or col < board_size-1
*/
void nextGeneration(bool* board, bool* next_board, int board_size){
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
