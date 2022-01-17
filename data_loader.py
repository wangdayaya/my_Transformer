import argparse
import os
import jieba
from hbconfig import Config
from pandas import np
from tqdm import tqdm
import tensorflow as tf

class IteratorInitializerHook(tf.train.SessionRunHook):
    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None
    def after_create_session(self, session, coord):
        self.iterator_initializer_func(session)

def get_dataset_batch(data, buffer_size = 10000, batch_size=64, scope='train'):
    iterator_initializer_hook = IteratorInitializerHook()
    def inputs():
        with tf.name_scope(scope):
            nonlocal data
            enc_inputs, targets = data
            enc_placeholder = tf.placeholder(tf.int32, [None, None], name='enc_placeholder')
            target_placeholder = tf.placeholder(tf.int32, [None, None], name='target_placeholder')

            dataset = tf.data.Dataset.from_tensor_slices((enc_placeholder, target_placeholder))
            if scope=='train':
                dataset = dataset.repeat(None)
            else:
                dataset = dataset.repeat(1)

            dataset = dataset.shuffle(buffer_size = buffer_size)
            dataset = dataset.batch(batch_size)

            iterator = dataset.make_initializable_iterator()
            next_enc, next_target = iterator.get_next()

            tf.identity(next_enc[0], 'enc_0')
            tf.identity(next_target[0], 'target_0')

            iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(iterator.initializer, feed_dict={enc_placeholder: enc_inputs, target_placeholder: targets})

            return {"enc_inputs": next_enc}, next_target

    return inputs, iterator_initializer_hook

def set_max_seq_length(fnames):
    Config.data.max_seq_length = 10
    for fname in fnames:
        input_data = open(os.path.join(Config.data.base_path, Config.data.processed_path, fname),'r')
        for line in input_data.readlines():
            Config.data.max_seq_length = max(Config.data.max_seq_length, len(line.split()))
    print(f"setting max_seq_length to Config : {Config.data.max_seq_length}")


def _pad_input(ids, max_seq_length):
    return ids + [Config.data.PAD_ID] * (max_seq_length- len(ids))


def load_data(enc_fname, dec_fname):
    enc_input_data = open(os.path.join(Config.data.base_path, Config.data.processed_path, enc_fname), 'r')
    dec_input_data = open(os.path.join(Config.data.base_path, Config.data.processed_path, dec_fname), 'r')
    
    enc_data, dec_data = [], []
    for e_line, d_line in tqdm(zip(enc_input_data.readlines(), dec_input_data.readlines())):
        e_ids = [int(id_) for id_ in e_line.split()]
        d_ids = [int(id_) for id_ in d_line.split()]
        if len(e_ids) == 0 or len(d_ids) == 0:
            continue
        if len(e_ids) <= Config.data.max_seq_length and len(d_ids) < Config.data.max_seq_length:
            enc_data.append(_pad_input(e_ids, Config.data.max_seq_length))
            dec_data.append(_pad_input(d_ids, Config.data.max_seq_length))

    print(f'load data from {enc_fname}, {dec_fname}')
    return np.array(enc_data, dtype=np.int32), np.array(dec_data, dtype=np.int32)

def make_train_and_test_set(shuffle=True):
    print('make Training data and Test data Start ...')
    if  Config.data.get('max_seq_length', None) is None:
        set_max_seq_length(['train_ids.enc','train_ids.dec','test_ids.enc','test_ids.dec'])

    train_enc, train_dec = load_data('train_ids.enc', 'train_ids.dec')
    test_enc, test_dec = load_data('test_ids.enc', 'test_ids.dec')

    assert len(train_enc) == len(train_dec)
    assert len(test_enc) == len(test_dec)

    print(f"train data count: {len(train_dec)}")
    print(f"test data count: {len(test_dec)}")

    if shuffle:
        print("shuffle dataset ...")
        train_p = np.random.permutation(len(train_dec))
        test_p = np.random.permutation(len(test_dec))
        return ((train_enc[train_p], train_dec[train_p]),(test_enc[test_p], test_dec[test_p]))
    else:
        return ((train_enc, train_dec),(test_enc, test_dec))

def make_dir(path):
    try:
        os.mkdir(path)
    except:
        pass

def build_vocab(en, de):
    def count_vocab(filePath):
        vocab = {}
        with open(filePath, 'rb') as f:
            for line in tqdm(f.readlines()):
                line = line.decode('utf-8')
                for token in jieba.cut(line.strip()):
                    if not token in vocab:
                        vocab[token] = 0
                    vocab[token] += 1
        return vocab

    def write_vocab(fileName, sorted_vocab):
        dest_path = os.path.join(Config.data.base_path, Config.data.processed_path, fileName)
        with open(dest_path, 'wb') as f:
            f.write(('<pad>' + '\n').encode('utf-8'))
            f.write(('<unk>' + '\n').encode('utf-8'))
            f.write(('<s>' + '\n').encode('utf-8'))
            f.write(('<\s>' + '\n').encode('utf-8'))
            for word,count in tqdm(sorted_vocab):
                if count < Config.data.word_threshold:
                    break
                f.write((word + '\n').encode('utf-8'))

    en_path = os.path.join(Config.data.base_path, Config.data.raw_data_path, en)
    de_path = os.path.join(Config.data.base_path, Config.data.raw_data_path, de)
    source_vocab = count_vocab(en_path)
    target_vocab = count_vocab(de_path)
    print("chinese vocab size: %d , english vocab size: %d"%(len(source_vocab), len(target_vocab)))

    sorted_source_vocab = sorted(source_vocab.items(), key=lambda x:x[1], reverse=True)
    sorted_target_vocab = sorted(target_vocab.items(), key=lambda x: x[1], reverse=True)
    write_vocab("source_vocab", sorted_source_vocab)
    write_vocab("target_vocab", sorted_target_vocab)


def load_vocab(vocab_path):
    result = {}
    with open(os.path.join(Config.data.base_path, Config.data.processed_path, vocab_path), 'rb') as f:
        words = f.readlines()
        for idx,word in enumerate(words):
            word = word.decode('utf-8').strip('\n')
            result[word] = idx
        print('%s vocab size: %d'%(vocab_path, len(words)))
    return result

def sentence2id(vocab, line):
    result = []
    for token in jieba.cut(line.strip()):
        result.append(vocab.get(token, vocab['<unk>']))
    return result


def token2id(data, mode):
    vocab_path = 'vocab'
    if mode == 'enc':
        vocab_path = 'source_' + vocab_path
    else:
        vocab_path = 'target_' + vocab_path
    vocab = load_vocab(vocab_path)
    
    in_path = data + '.' + mode
    out_path = data + '_ids.' + mode
    in_file = open(os.path.join(Config.data.base_path, Config.data.raw_data_path, in_path), 'rb')
    out_file = open(os.path.join(Config.data.base_path, Config.data.processed_path, out_path), 'wb')

    lines = in_file.readlines()
    for line in tqdm(lines):
        ids = []
        sentence_ids = sentence2id(vocab, line.strip().decode('utf-8'))
        ids.extend(sentence_ids)
        if mode == 'dec':
            ids.append(vocab['<\s>'])
            ids.append(vocab['<pad>'])
        s = ' '.join(str(id) for id in ids) + '\n'
        out_file.write(s.encode('utf-8'))

def process_data():
    print('preparing data ...')
    make_dir(os.path.join(Config.data.base_path, Config.data.processed_path))
    build_vocab('train.enc','train.dec')
    token2id('train', 'enc')
    token2id('train', 'dec')
    token2id('test', 'enc')
    token2id('test', 'dec')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str, default='Hyperparameter', help='config file name')
    args = parser.parse_args()
    Config(args.config)
    process_data()

