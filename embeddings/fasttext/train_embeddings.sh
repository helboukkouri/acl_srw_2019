mkdir pretrained/

CORPORA_PATH="../corpora"
for corpus in $(find ${CORPORA_PATH} -maxdepth 1 -mindepth 1 -type d -printf '%f\n')
do
    CORPUS="${CORPORA_PATH}/$corpus/corpus.txt"

    mkdir pretrained/$corpus/

    SAVE_FILE=pretrained/$corpus/vectors
    
    BUILDDIR='fastText-0.2.0'
    
    VECTOR_SIZE=256
    VOCAB_MIN_COUNT=5
    WINDOW_SIZE=5
    NEGATIVE_SAMPLES=5
    MIN_NGRAM=3
    MAX_NGRAM=6
    MAX_ITER=10
    
    NUM_THREADS=30

    ./$BUILDDIR/fasttext skipgram \
        -input "${CORPUS}" \
        -output "${SAVE_FILE}" \
        -lr 0.025 \
        -dim $VECTOR_SIZE \
        -ws $WINDOW_SIZE \
        -epoch $MAX_ITER \
        -minCount $VOCAB_MIN_COUNT \
        -neg $NEGATIVE_SAMPLES \
        -loss ns \
        -bucket 2000000 \
        -minn $MIN_NGRAM \
        -maxn $MAX_NGRAM \
        -thread $NUM_THREADS \
        -t 1e-4 \
        -lrUpdateRate 100
done
