FastNerualNets
==============

Fast C++ implementation of deep neural networks. The project is under active development by Boris Vidolov. Please come back soon, as the library is updated daily.

##Current state:##
 Building template classes created for the network.<br/>
 Support for double precision floating point numbers only.<br/>
 AVX implementation for faster calculation of forward network propagation (some 2x faster calculation)<br/>
 OMP implementation for batch calculations (additionally NUM_CPU_CORES times faster)<br/>
 Training with genetic algorithms<br/>
 <br/>
 Next steps:<br/>
 1. Implement backprop<br/>
 2. Contrastive divergence<br/>
 3. OpenCL execution support<br/>

