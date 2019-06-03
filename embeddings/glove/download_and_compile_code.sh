# Download word2vec source code
wget https://nlp.stanford.edu/software/GloVe-1.2.zip

# Unpack files and remove archive
unzip GloVe-1.2.zip
rm GloVe-1.2.zip

# Compile source code
cd GloVe-1.2/
make clean
make
cd ..
