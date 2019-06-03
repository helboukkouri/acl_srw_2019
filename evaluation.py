import os
import re
import shutil
import subprocess

from config import I2B2_PATH

def space_tokenizer(text):
    tokens = re.split('\ +', text)

    spans = []
    start = 0
    for token in tokens:
        if token == '':
            spans.append((start, start))
        else:
            
            this_span = re.search(re.escape(token), text[start:])
            assert this_span is not None
            
            this_span = this_span.span()
            spans.append((this_span[0] + start, this_span[1] + start))
            
            start += this_span[1]

    return tokens, spans


def convert_i2b2_format(all_iob_sequences, output_path, sort_files=True,
        test_texts_path=os.path.join(I2B2_PATH, '2010/test/texts/')):

    os.makedirs(output_path, exist_ok=True)

    txt_filenames = os.listdir(test_texts_path)
    if sort_files:
        txt_filenames.sort()

    for filename in txt_filenames:

        text = open(test_texts_path + filename, 'r', encoding='utf-8-sig').read()

        sentences = text.split('\n')
        token_sequences_with_spans = [space_tokenizer(sentence) for sentence in sentences]

        sentence_id = 0
        annotations = []
        for token_sequence, span_sequence in token_sequences_with_spans:

            iob_sequence = all_iob_sequences.pop(0)
            iob_sequence = list(iob_sequence)
            if len(iob_sequence) != len(token_sequence):
                if len(iob_sequence) + 1 == len(token_sequence) and token_sequence[-1] == '':
                    iob_sequence += ['O']
                elif len(iob_sequence) == len(token_sequence) + 1 and iob_sequence[-1] == 'O':
                    iob_sequence.pop(-1)
                else:
                    print('Lengths don\'t match:')
                    print(iob_sequence, token_sequence)
                    raise ValueError

            last_iob_tag_type = None
            last_iob_tag_position = None

            last_span_start = 0
            last_span_end = 0
            last_id_start = 0
            last_id_end = 0

            token_id = 0
            for token, span in zip(token_sequence, span_sequence):

                iob_tag = iob_sequence.pop(0)
                if iob_tag == 'O':
                    iob_tag_position = ''
                    iob_tag_type = 'O'
                else:                
                    iob_tag_position = iob_tag.split('-')[0]
                    iob_tag_type = iob_tag.split('-')[1]

                if last_iob_tag_type != iob_tag_type \
                    or (last_iob_tag_position == 'B' and iob_tag_position == 'B'):

                    if last_iob_tag_type != 'O' and last_iob_tag_type is not None:
                        annotations.append(
                            [sentence_id,
                             last_span_start, last_span_end,
                             last_id_start, last_id_end,
                             last_iob_tag_type])

                    last_iob_tag_type = iob_tag_type
                    last_iob_tag_position = iob_tag_position

                    last_span_start = span[0]
                    last_span_end = span[1]
                    last_id_start = token_id
                    last_id_end = token_id
                    #last_concept_id = concept_id
                else:
                    last_span_end = span[1]
                    last_id_end = token_id

                token_id += 1

            if last_iob_tag_type != 'O' and last_iob_tag_type is not None:
                annotations.append(
                    [sentence_id,
                     last_span_start, last_span_end,
                     last_id_start, last_id_end,
                     last_iob_tag_type])

            sentence_id += 1

        with open(os.path.join(output_path, filename[:-3] + 'con'), 'w') as writer:
            for ann in annotations:
                if ann[5]:
                    text = 'c="'
                    text += re.sub('\ +', ' ', sentences[ann[0]][ann[1]:ann[2]]).lower()
                    text += '" '
                    text += str(ann[0] + 1) + ':' + str(ann[3]) + ' ' + str(ann[0] + 1) + ':' + str(ann[4])
                    text += '||t="'
                    text += ann[5]
                    text += '"\n'

                    # print(text)
                    writer.write(text)


def i2b2_evaluation(all_iob_sequences, output_path, logger=None, verbose=True, sort_files=True,
        i2b2_path=I2B2_PATH):

    test_concepts_path = i2b2_path + '2010/test/concepts/'
    test_texts_path = i2b2_path + '2010/test/texts/'
    eval_jar = 'i2b2va-eval.jar'
    
    convert_i2b2_format(all_iob_sequences, output_path=output_path,
                        test_texts_path=test_texts_path, sort_files=sort_files)

    cmd = 'java -jar %s -rcp %s -scp %s -ft con -ex all' % (eval_jar, test_concepts_path, output_path)
    status = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    result = status.stdout.decode()

    if verbose:
        if logger:
            logger.info(result)
            logger.info("----====----\n\n")
            logger.info(result.split('\n')[7])
        else:
            print(result)
            print("----====----\n\n")
            print(result.split('\n')[7])

    shutil.rmtree(output_path)

    return result.split('\n')[7].split()[-3:]
