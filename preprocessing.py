# based on ideas from https://github.com/dennybritz/cnn-text-classification-tf

import numpy as np
import codecs
import json
#import jieba
import string

CHINESE_SPAM = ''
CHINESE_HAM = ''
FINAL_EMBEDDINGS = 'E:/tmp/word-embeddings/glove.6B.100d.txt'
YELP_700K = 'E:/tmp/yelp/yelp_700K.json'
MR_POS = 'E:/tmp/movie-review/rt-polarity.pos'
MR_NEG = 'E:/tmp/movie-review/rt-polarity.neg'
sequence_length = 128

def load_yelp(alphabet):
    examples = []
    labels = []
    with codecs.open(YELP_700K, 'r', 'utf-8') as f:
        i = 0
        for line in f:
            review = json.loads(line)
            stars = review["stars"]
            text = review["text"]
            if stars != 3:
                text_end_extracted = extract_end(list(text.lower()))
                padded = pad_sentence(text_end_extracted)
                text_int8_repr = string_to_int8_conversion(padded, alphabet)
                #negative=[1,0]
                if stars == 1 or stars == 2:
                    labels.append([1, 0])
                    examples.append(text_int8_repr)
                #positive=[0,1]
                elif stars == 5 or stars == 4:
                    labels.append([0, 1])
                    examples.append(text_int8_repr)
                i += 1
                if i % 10000 == 0:
                    print("Non-neutral instances processed: " + str(i))
                if i >= 330000:
                    break
    return examples, labels

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    import re
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

#for yelp data w2v model
def load_yelp_w2v(lookup_table, feature_length):
    contents = []
    labels =[]
    import re
    #from random import randint
    with codecs.open(YELP_700K, 'r', 'utf-8') as f:
        i = 0
        for line in f:
            i += 1
            fine_grained = False
            #rd = randint(0, 99)
            if not fine_grained:
                features = []
                review = json.loads(line)
                stars = review["stars"]
                text = review["text"]
                padding = ['0.0'] * 100
                if stars != 3:

                    cleaned_str = clean_str(text)
                    #spaced = re.sub(r"[%s]+" % string.punctuation, " ", text)
                    seg_list = cleaned_str.split()

                    for item in seg_list:
                        if item in lookup_table:
                            features.append(lookup_table[item])
                        else:
                            features.append(lookup_table['unk'])
                            #features.append(padding)

                    if len(features) >= feature_length:
                        result = features[:feature_length]
                    else:
                        num_padding = feature_length - len(features)
                        result = features + [lookup_table['unk']] * num_padding
                        #result = features + padding * num_padding

                    if stars == 1 or stars == 2:
                        labels.append([1, 0])
                    elif stars == 5 or stars == 4:
                        labels.append([0, 1])

                    arr = np.asarray(result, dtype=np.float32).transpose()
                    contents.append(arr)


                    if len(labels) % 5000 == 0:
                        print("%d non-neutral instances loaded..." % len(labels))
                    if len(labels) >= 50000:
                        break
            else:
                features = []
                review = json.loads(line)
                stars = review["stars"]
                text = review["text"]

                text = text.lower().replace("\n", " ")
                spaced = re.sub(r"[%s]+" % string.punctuation, " ", text)
                seg_list = spaced.split()

                for item in seg_list:
                    if item in lookup_table:
                        features.append(lookup_table[item])
                    else:
                        features.append(lookup_table['UNK'])

                if len(features) >= feature_length:
                        result = features[:feature_length]
                else:
                    num_padding = feature_length - len(features)
                    result = features + [lookup_table['UNK']] * num_padding

                if stars == 1:
                    labels.append([1, 0, 0, 0, 0])
                elif stars == 2:
                    labels.append([0, 1, 0, 0, 0])
                elif stars == 3:
                    labels.append([0, 0, 1, 0, 0])
                elif stars == 4:
                    labels.append([0, 0, 0, 1, 0])
                else:
                    labels.append([0, 0, 0, 0, 1])

                arr = np.asarray(result, dtype=np.float32).transpose()
                contents.append(arr)

                if len(labels) % 5000 == 0:
                    print("%d instances loaded..." % len(labels))
                if len(labels) >= 50000:
                    break

    return contents, labels
    
def load_spam_data(alphabet):
    contents = []
    labels = []
    with codecs.open(CHINESE_HAM, 'r', 'utf-8') as f1:
        i = 0;
        for line in f1:
            text_start_extracted = extract_start(list(line.lower()))
            padded = pad_sentence(text_start_extracted)
            text_int8_repr = string_to_int8_conversion(padded, alphabet)
            contents.append(text_int8_repr)
            labels.append([0, 1])
            
    with codecs.open(CHINESE_SPAM, 'r', 'utf-8') as f2:
            for line in f2:
                text_start_extracted = extract_start(list(line.lower()))
                padded = pad_sentence(text_start_extracted)
                text_int8_repr = string_to_int8_conversion(padded, alphabet)
                contents.append(text_int8_repr)
                #spam=[1,0]
                labels.append([1, 0])

    return contents, labels

def load_mr(lookup_table, feature_length):
    print("Loading movie review data...")

    contents = []
    labels = []
    import re
    with codecs.open(MR_POS, 'r', 'utf-8') as f1:
        i = 0;
        for line in f1:
            features = []
            cleaned_str = clean_str(line)
            #spaced = re.sub(r"[%s]+" % string.punctuation, " ", cleaned_str)
            #seg_list = spaced.split()
            seg_list = cleaned_str.split()

            for item in seg_list:
                if item in lookup_table:
                    features.append(lookup_table[item])
                else:
                    features.append(lookup_table['unk'])

            if len(features) >= feature_length:
                result = features[:feature_length]
            else:
                num_padding = feature_length - len(features)
                result = features + [lookup_table['unk']] * num_padding
            arr = np.asarray(result, dtype=np.float32).transpose()
            contents.append(arr)

            labels.append([0, 1])

    with codecs.open(MR_NEG, 'r', 'utf-8') as f2:
        for line in f2:
            features = []
            cleaned_str = clean_str(line)
            #spaced = re.sub(r"[%s]+" % string.punctuation, " ", cleaned_str)
            #seg_list = spaced.split()
            seg_list = cleaned_str.split()

            for item in seg_list:
                if item in lookup_table:
                    features.append(lookup_table[item])
                else:
                    features.append(lookup_table['unk'])

            if len(features) >= feature_length:
                result = features[:feature_length]
            else:
                num_padding = feature_length - len(features)
                result = features + [lookup_table['unk']] * num_padding
            arr = np.asarray(result, dtype=np.float32).transpose()
            contents.append(arr)

            labels.append([1, 0])

    return contents, labels

#for chinese spam w2v model
def load_spam_w2v(lookup_table, feature_length):
    pass
    '''
    from zhon.hanzi import punctuation
    import re
    contents = []
    labels = []
    count = 0;
    with codecs.open('./data_spam/ham_merged.txt', 'r', 'utf-8') as f1:
        for line in f1:
            features = []
            line = re.sub(r"[%s]+" % punctuation, "", line)
            seg_list = jieba.cut(line, cut_all=False)
            for item in seg_list:
                if item in lookup_table:
                    features.append(lookup_table[item])
                else:
                    features.append(lookup_table['UNK'])

            if len(features) >= feature_length:
                result = features[:feature_length]
            else:
                num_padding = feature_length - len(features)
                result = features + [lookup_table['UNK']] * num_padding

            arr = np.asarray(result, dtype=np.float32).transpose()
            contents.append(arr)
            labels.append([0, 1])
            count += 1
            if count % 2000 == 0:
                print("%d lines of data loaded..." % len(labels))
            
    with codecs.open('./data_spam/spam_merged.txt', 'r', 'utf-8') as f2:
            for line in f2:
                features = []
                line = re.sub(r"[%s]+" % punctuation, "", line)
                seg_list = jieba.cut(line, cut_all=False)
                for item in seg_list:
                    if item in lookup_table:
                        features.append(lookup_table[item])
                    else:
                        features.append(lookup_table['UNK'])

                if len(features) >= feature_length:
                    result = features[:feature_length]
                else:
                    num_padding = feature_length - len(features)
                    result = features + [lookup_table['UNK']] * num_padding

                arr = np.asarray(result, dtype=np.float32).transpose()
                contents.append(arr)
                labels.append([1, 0])
                count += 1
                if count % 2000 == 0:
                    print("%d lines of data loaded..." % len(labels))

    return contents, labels
    '''
def build_lookup_table(filename):
    lookup_table = dict()
    with codecs.open(filename, "r", "utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            #split = line.split(",")
            #for glove vector
            split = line.split(" ")
            key = split[0]
            value = split[1:]
            lookup_table[key] = value
    print("finish building lookup_table, size:", len(lookup_table))
    return lookup_table

def extract_start(char_seq):
    if len(char_seq) > 1014:
        char_seq = char_seq[:1014]
    return char_seq

def extract_end(char_seq):
    if len(char_seq) > 1014:
        char_seq = char_seq[-1014:]
    return char_seq


def pad_sentence(char_seq, padding_char=" "):
    char_seq_length = 1014
    num_padding = char_seq_length - len(char_seq)
    new_char_seq = char_seq + [padding_char] * num_padding
    return new_char_seq


def string_to_int8_conversion(char_seq, alphabet):
    x = np.array([alphabet.find(char) for char in char_seq], dtype=np.int8)
    return x


def get_batched_one_hot(char_seqs_indices, labels, start_index, end_index):
    #alphabet = r"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\"/_@#$%>()[]{}*~"
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
    x_batch = char_seqs_indices[start_index:end_index]
    y_batch = labels[start_index:end_index]
    x_batch_one_hot = np.zeros(shape=[len(x_batch), len(alphabet), len(x_batch[0]), 1])
    for example_i, char_seq_indices in enumerate(x_batch):
        for char_pos_in_seq, char_seq_char_ind in enumerate(char_seq_indices):
            if char_seq_char_ind != -1:
                x_batch_one_hot[example_i][char_seq_char_ind][char_pos_in_seq][0] = 1
    return [x_batch_one_hot, y_batch]


def load_data():
    # TODO Add the new line character later for the yelp'cause it's a multi-line review
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
    examples, labels = load_yelp(alphabet)
    
    #alphabet = r"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\"/_@#$%>()[]{}*~"
    #examples, labels = load_spam_data(alphabet)
    
    x = np.array(examples, dtype=np.int8)
    y = np.array(labels, dtype=np.int8)
    print("x_char_seq_ind=" + str(x.shape))
    print("y shape=" + str(y.shape))
    return [x, y]

def load_data_w2v():
    lookup_table = build_lookup_table(FINAL_EMBEDDINGS)
    examples, labels = load_yelp_w2v(lookup_table, sequence_length)
    #reduce memory
    del lookup_table

    x = np.asarray(examples, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int8)
    print("x shape="+str(x.shape))
    print("y shape=" + str(y.shape))
    return [x, y]


def batch_iter(x, y, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    # data = np.array(data)
    data_size = len(x)
    num_batches_per_epoch = int(data_size/batch_size) + 1
    for epoch in range(num_epochs):
        print("In epoch >> " + str(epoch + 1))
        print("num batches per epoch is: " + str(num_batches_per_epoch))
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]
        else:
            x_shuffled = x
            y_shuffled = y
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            x_batch, y_batch = get_batched_one_hot(x_shuffled, y_shuffled, start_index, end_index)
            batch = list(zip(x_batch, y_batch))
            yield batch

def batch_iter_w2v(x, y, batch_size, num_epochs, w2v_dim, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    # data = np.array(data)
    data_size = len(x)
    num_batches_per_epoch = int(data_size/batch_size) + 1
    for epoch in range(num_epochs):
        print("In epoch >> " + str(epoch + 1))
        #print("num batches per epoch is: " + str(num_batches_per_epoch))
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]
        else:
            x_shuffled = x
            y_shuffled = y
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            x_batch = x_shuffled[start_index:end_index]
            x_batch = np.reshape(x_batch, [len(x_batch), w2v_dim, sequence_length, 1])
            y_batch = y_shuffled[start_index:end_index]
            batch = list(zip(x_batch, y_batch))
            yield batch
