git clone https://github.com/coccoc/coccoc-tokenizer
cd coccoc-tokenizer
mkdir build
cd build
cmake -DBUILD_PYTHON=1 -DCMAKE_INSTALL_PREFIX=$HOME/.local ..
make install
cd ../python
python setup.py install