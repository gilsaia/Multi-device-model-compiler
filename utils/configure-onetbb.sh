DIR=$(pwd)
mkdir 3rdparty/oneTBB/build
cd 3rdparty/oneTBB/build
cmake -G Ninja .. \
    -DTBB_TEST=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$DIR/3rdparty/oneTBB/build/tbb