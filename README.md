# CUDA programming Project

**Due date Friday 17 June at 11:30 pm**

## Instruction and important things to note

* Fork the project by clicking the "Fork" button on the top-right corner.
* **Make sure that the visibility of your new repository (the one you just forked) is set to private.**
* You can obtain the URL of your forked project by clicking the "Clone" button (beside "Fork") and then the “copy to clipboard icon" at the right of the dropdown.
* Commit your changes regularly, providing an informative commit message(Commit and Changes Tutorial: https://www.jetbrains.com/help/idea/commit-and-push-changes.html)
* You are expected to make at least 10 commits with messages to explain what have changed. 10 out of 50 marks are allocated for this. 
* **Please note, you choose to do task option 1 or option 2, but NOT both.**

## Your task: self-study CUDA programming and implement a simple CUDA program ##

To develop and run a CUDA program, you will need to use a computer with NVIDIA GPUs and have the [CUDA development toolkit] (https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) installed, or you can use the Lab 2 machines where The CUDA development toolkit has already been installed in  
```/home/compx553/cuda```

To use it, in your home directory open _.profile_ file (please note it is a hidden file, so you might need to switch on "show hidden files" to see it), and add the following lines to the end of file. Save the file and run ```source .profile``` in a teminal.

```
export CUDA_PATH=/home/compx553/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64
```

**Please note,  this option will only give you upto 5/100 marks, so only choose it when you run out of time or stuggle with the task option 2** 


## Task Option 1 (5%)
Your task is to reimplement the throwing darts assignment using CUDA and write a  brief report to map the terminologies used by OpenCL and CUDA.



## Task Option 2 (10%)

Your task is to implement the [Game of Life](http://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) using CUDA, and to see how much speedup you can get on a GPU.

    






