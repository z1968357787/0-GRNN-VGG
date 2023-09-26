# paths
mode = 'pred'
qa_path = '/home/xie/下载/VQA-all/VQA2/'

preprocessed_trainval_path = '/home/cike/faster-rcnn.pytorch-pytorch-1.0/thekbdata_x.h5'#'/home/cike/VQA2.0-Recent-Approachs-2018.pytorch-master/faster-rcnn.pytorch-pytorch-1.0/x_genome-trainval.h5'#'/home/xie/下载/VQA-all/VQG-code/x_genome-trainval.h5'  # path where preprocessed features from the trainval split are saved to and loaded from

vocabulary_path = 'data/VQA2_vocab.json'

train_answers_path = 'defect-detect-json/train.json'
train_questions_path = 'defect-detect-json/train.json'

valid_questions_path = 'defect-detect-json/valid.json'
valid_answers_path = 'defect-detect-json/valid.json'





glove_index = '../data/dictionary.pkl'
embedding_path = '../data/glove6b_init_300d.npy'
glove_emc = '/home/cike/VQA2.0-Recent-Approachs-2018.pytorch-master/VQG/word_embedding.pkl'
min_word_freq = 3
max_q_length = 666 # question_length = min(max_q_length, max_length_in_dataset)
max_a_length = 4

max_o_length = 20
batch_size = 16
data_workers = 4
normalize_box = True

seed = 2020
