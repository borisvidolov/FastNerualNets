FastNeuralNets
==============

Fast C++ implementation of deep neural networks. The project is under active development by Boris Vidolov. Please come back soon, as the library is updated daily.

##Current state:##
 Building template classes created for the network.<br/>
 Support for double precision floating point numbers only.<br/>
 AVX implementation for faster calculation of forward network propagation (~ 2x faster calculation)<br/>
 OMP implementation for batch calculations (additionally NUM_CPU_CORES times faster)<br/>
 Training with genetic algorithms (2x faster than for training XOR than back propagation)<br/>
 Backpropagation is implemented, but could be optimized further.<br/>
 <br/>
##Next steps:##
 1. Contrastive divergence<br/>
 2. OpenCL execution support<br/>
 3. OpenCL hooked up with genetic algorithms<br/>
 3. OpenCL training<br/>

