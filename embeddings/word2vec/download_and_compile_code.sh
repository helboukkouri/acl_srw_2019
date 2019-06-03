# Download word2vec source code
wget https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip

# Unpack files and remove archive
unzip source-archive.zip
rm source-archive.zip

# Compile source code
cd word2vec/trunk/
make clean
make
cd ../..
