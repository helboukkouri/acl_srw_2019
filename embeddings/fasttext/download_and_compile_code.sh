# Download word2vec source code
wget https://github.com/facebookresearch/fastText/archive/v0.2.0.zip

# Unpack files and remove archive
unzip v0.2.0.zip
rm v0.2.0.zip

# Compile source code
cd fastText-0.2.0
make clean
make
cd ..
