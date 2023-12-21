cd protobuf
./autogen.sh
./configure --enable-static=no
make -j$(sysctl -n hw.logicalcpu) install
cd python
python3 setup.py install --cpp_implementation
