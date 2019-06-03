mkdir pretrained/

CORPORA_PATH="../corpora"
for corpus in $(find ${CORPORA_PATH} -maxdepth 1 -mindepth 1 -type d -printf '%f\n')
do
    CORPUS="${CORPORA_PATH}/$corpus/corpus.txt"

    mkdir pretrained/$corpus/

    SAVE_FILE=pretrained/$corpus/vectors.bin
    VOCAB_FILE=pretrained/$corpus/vocabulary.txt
    
    BUILDDIR='word2vec/trunk'
    
    CBOW=1
    VECTOR_SIZE=256
    VOCAB_MIN_COUNT=5
    MAX_ITER=10
    WINDOW_SIZE=5
    
    NUM_THREADS=30
    BINARY=1

    time ./$BUILDDIR/word2vec \
        -train "${CORPUS}" \
        -output "${SAVE_FILE}" \
        -save-vocab "${VOCAB_FILE}" \
        -binary $BINARY \
        -size $VECTOR_SIZE \
        -window $WINDOW_SIZE \
        -min-count $VOCAB_MIN_COUNT \
        -cbow $CBOW \
        -iter $MAX_ITER \
        -threads $NUM_THREADS
done
