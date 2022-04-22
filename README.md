# CUDA programming Project

**Due date Friday 17 June at 11:30 pm**

## Instruction and important things to note

* Fork the project by clicking the "Fork" button on the top-right corner.
* **Make sure that the visibility of your new repository (the one you just forked) is set to private.**
* You can obtain the URL of your forked project by clicking the "Clone" button (beside "Fork") and then the “copy to clipboard icon" at the right of the dropdown.
* Commit your changes regularly, providing an informative commit message(Commit and Changes Tutorial: https://www.jetbrains.com/help/idea/commit-and-push-changes.html)
* You are expected to make at least 10 commits with messages to explain what have changed. 10 out of 50 marks are allocated for this. 


## Your task: self-study CUDA programming and implement a simple CUDA program ##

To develop and run a CUDA program, you will need to use a computer with NVIDIA GPUs and have the CUDA development toolkit (https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) installed, or you can use the Lab 2 machines where The CUDA development toolkit has already been installed in  ```/home/compx553/cuda```

To use it, in your home directory open _.profile_ file (please note it is a hidden file, so you might need to switch on "show hidden files" to see it), and add the following lines to the end of file. Save the file and run ```source .profile``` 
in a teminal.

```
export CUDA_PATH=/home/compx553/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64
```

### Useful resources ###

Here are the resources I have used when studying CUDA programming and doing the project myself, which might help get you started.

* basics: https://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf
* tutorials: https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/
* CUDA programming guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
* a blog on how to use CUDA unified memory: https://developer.nvidia.com/blog/even-easier-introduction-cuda/
* a blog on CUDA's execution and memory model: https://jhui.github.io/2017/03/06/CUDA/

### Task Option 1 (5%)
The task is to reimplement the throwing darts assignment using CUDA and write a  brief report to map the terminologies used by OpenCL and CUDA. **Please note, you either choose this option or the option 2 below, NOT both. This option will only give you upto 5/100 marks, so choose it ony when you run out of time or stuggle with the task option 2** 

### Task Option 2 (10%)

The task is to implement the Conway's Game of Life(http://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) using CUDA, and to see how much speedup you can get on a GPU. A template file is provided to you.


### report ###
you must also deliver a short report (i.e. a pdf document) that records the run time for each generation on CPU and GPU (only for option 2) and a table that containing the mapping of OpenCL and CUDA terminologies as examples given below below. 


|OpenCL|CUDA| Note |
|-----|--------|-------------------|
|work item | thread|
|word group |  | |
|word group |  | |     




## Grading for task option 1 (total 5 marks)

|Marks|Allocated to|
|-----|--------|
|2 | implementation|
|3 | report | 


## Grading for task option 2 (total 10 marks)

|Marks|Allocated to|
|-----|--------|
|5 | implementation|
|2 | caching a region of cells using shared memory| 
|3 | report | 

