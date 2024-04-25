#include <iostream>
#include <random>
#include <chrono>
#include <ctime>

using std::cout; using std::endl;


void init_board(bool* board, int board_size){
}



void print_board(bool *board, int board_size, int range){
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
