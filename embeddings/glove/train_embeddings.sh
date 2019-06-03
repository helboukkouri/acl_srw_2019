mkdir pretrained/

CORPORA_PATH="../corpora"
for corpus in $(find ${CORPORA_PATH} -maxdepth 1 -mindepth 1 -type d -printf '%f\n')
do
    CORPUS="${CORPORA_PATH}/$corpus/corpus.txt"

    mkdir pretrained/$corpus/

    SAVE_FILE=pretrained/$corpus/vectors
    VOCAB_FILE=pretrained/$corpus/vocabulary.txt
    COOCCURRENCE_FILE=pretrained/$corpus/cooccurrence.bin
    COOCCURRENCE_SHUF_FILE=pretrained/$corpus/cooccurrence.shuf.bin
    
    BUILDDIR='GloVe-1.2/build'
    
    VECTOR_SIZE=256
    VOCAB_MIN_COUNT=5
    MAX_ITER=10
    WINDOW_SIZE=15
    
    NUM_THREADS=30
    VERBOSE=2
    BINARY=2

    ./$BUILDDIR/vocab_count \
        -min-count $VOCAB_MIN_COUNT \
        -verbose $VERBOSE \
        < $CORPUS > $VOCAB_FILE
    
    ./$BUILDDIR/cooccur \
        -vocab-file $VOCAB_FILE \
        -verbose $VERBOSE \
        -window-size $WINDOW_SIZE \
        < $CORPUS > $COOCCURRENCE_FILE
    
    ./$BUILDDIR/shuffle \
        -verbose $VERBOSE \
        < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
    
    ./$BUILDDIR/glove \
        -save-file $SAVE_FILE \
        -threads $NUM_THREADS \
        -input-file $COOCCURRENCE_SHUF_FILE \
        -iter $MAX_ITER \
        -vector-size $VECTOR_SIZE \
        -binary $BINARY \
        -vocab-file $VOCAB_FILE \
        -verbose $VERBOSE
done
