# SOFIE
This is an experimental standalone version of SOFIE - a tool for Fast ML Inference within ROOT - the scientific data analysis framework.

Since SOFIE is a part of ROOT and therefore needs to be built altogether, it takes quite a long time in its development and testing. This standalone version allows you to just build SOFIE with the pre-built binaries of ROOT- making the entire development process way faster.


## Installation

1. Getting a ROOT binary.  
Download a pre-built binary of ROOT based on your architecture from [here](https://root.cern/install/).

2. Build standalone SOFIE
```bash
git clone https://github.com/sanjibansg/SOFIE.git
cd SOFIE
mkdir build && cd build
cmake -Dtesting=ON -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
cmake --build . --target install -j10
```
The commands above should build the SOFIE standalone. To include it within the ROOT binary and run altogether, we need to source the shared libraries for `SOFIE_core` and `SOFIE_parsers`. Within the SOFIE repository we may call

```bash
source setup.sh

```
Now ROOT should also access the SOFIE libraries while it runs. This helps to accelerate development. Submit your developments here and we will proceed with the developments in ROOT carefull.


    
## Other Common Github Profile Sections
The standalone version of SOFIE is developed with inspiration from the standalone version of RooFit developed by Jonas Rembser that can be found [here](https://github.com/guitargeek/roofit).
