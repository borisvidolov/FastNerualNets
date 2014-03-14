FastNeuralNets
==============

Fast C++ implementation of deep neural networks. The project is under active development by Boris Vidolov. Please come back soon, as the library is updated daily.

##Current state:##
 Building template classes created for the network.<br/>
 Support for double precision floating point numbers only.<br/>
 AVX implementation for faster calculation of forward network propagation (some 2x faster calculation)<br/>
 OMP implementation for batch calculations (additionally NUM_CPU_CORES times faster)<br/>
 Training with genetic algorithms<br/>
 Backpropagation is implemented. However, at this stage, the algorithm signifficuntly underporfrms the genetic algorithms. This is not expected, so I am still tweeking the algorithm to converge to the local (and ideally global) minimum.<br/>
 <br/>
 Next steps:<br/>
 1. Fix backprop<br/>
 2. Contrastive divergence<br/>
 3. OpenCL execution support<br/>
 4. OpenCL training<br/>

