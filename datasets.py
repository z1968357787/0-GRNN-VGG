import json
import os
import os.path
import re
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# import _pickle as cPickle
from PIL import Image
import h5py
import torch
import torch.utils.data as data
import numpy as np
from collections import Counter
import config
from torchvision import transforms

preloaded_vocab = None

def get_img(img_id):
    # train_folder = '/home/cike/project/MO-VQG/source/VQA2.0/unzip/train2014'
    train_folder = 'defect-detect-img/train'
    valid_folder = 'defect-detect-img/valid'

    # train_path = 'COCO_train2014_' + str(img_id).zfill(12) + '.png'
    train_path = str(img_id) + '.jpg'
    valid_path = str(img_id) + '.jpg'
    #print(train_path)
    if train_path in os.listdir(train_folder):
        return os.path.join(train_folder, train_path)
    if valid_path in os.listdir(valid_folder):
        return os.path.join(valid_folder, valid_path)
    print(' Not Found ')
    return 'Not Found'

def get_loader(mode, features_file):
    """ Returns a data loader for the desired split """
    if mode == 'train':
        question_path = config.train_questions_path
        answer_path = config.train_answers_path
    elif mode == 'valid':
        question_path = config.valid_questions_path
        answer_path = config.valid_answers_path
    elif mode == 'test':
        question_path = config.test_questions_path
        answer_path = config.test_answers_path
        # question_path = config.valid_questions_path
        # answer_path = config.valid_answers_path
        config.batch_size = 1
    split = VQG(
        question_path,
        answer_path,
        features_file,
        answerable_only=True if mode == 'train' else False,
    )
    loader = torch.utils.data.DataLoader(
        split,
        batch_size=config.batch_size,
        shuffle=True if mode == 'train' else False,  # only shuffle the data in training
        pin_memory=True,
        num_workers=config.data_workers,
        # collate_fn=collate_fn,
    )
    return loader


def collate_fn(batch):
    # put question lengths in descending order so that we can use packed sequences later
    batch.sort(key=lambda x: x[-1], reverse=True)
    return data.dataloader.default_collate(batch)


class VQG(data.Dataset):
    """ VQA dataset, open-ended """
    def __init__(self, questions_path, answers_path, features_file, answerable_only=False):
        super(VQG, self).__init__()
        with open(questions_path, 'r') as fd:
            questions_json = json.load(fd)
        with open(answers_path, 'r') as fd:
            answers_json = json.load(fd)
        if preloaded_vocab:
            vocab_json = preloaded_vocab
        else:
            with open(config.vocabulary_path, 'r') as fd:
                vocab_json = json.load(fd)

        self.question_ids = [q['nlq_idx'] for q in questions_json]
 #questions_json['nlq_idx'] # [q['question_id'] for q in questions_json['questions']]

        # vocab
        self.vocab = vocab_json
        self.token_to_index = self.vocab
        self.answer_to_index = self.vocab
        # self.difficult_dict = {'easy': 0, 'difficult': 1}
        #self.difficult_dict = {'easy': 1, 'difficult': 0} # reverse

        # q and a
        self.questions = list(prepare_questions(questions_json))
        self.answers = list(prepare_answers(answers_json))
        self.questions = [self._encode_question(q) for q in self.questions]
        self.answers = [self._encode_answers(a) for a in self.answers]
        #self.difficult_level = [self.difficult_dict[q['difficult_level']] for q in questions_json['questions']]
        #print(self.difficult_dict)
        #print(self.difficult_level[:5])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # v
        self.image_features_path = features_file
        #self.coco_id_to_index = self._create_coco_id_to_index()
        self.coco_ids = [q['img_id'] for q in questions_json]
        # print(self.image_features[image_id])

        # only use questions that have at least one answer?
        self.answerable_only = answerable_only
        if self.answerable_only:
            self.answerable = self._find_answerable(not self.answerable_only)

    @property
    def max_question_length(self):
        if not hasattr(self, '_max_length'):
            data_max_length = max(map(len, self.questions))
            self._max_length = min(config.max_q_length, data_max_length)
        return self._max_length

    @property
    def num_tokens(self):
        return len(self.token_to_index)

    def _create_coco_id_to_index(self):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
        with h5py.File(self.image_features_path, 'r') as features_file:
            coco_ids = features_file['ids'][()]
        coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
        return coco_id_to_index

    def _check_integrity(self, questions, answers):
        """ Verify that we are using the correct data """
        qa_pairs = list(zip(questions['questions'], answers['annotations']))
        assert all(q['question_id'] == a['question_id'] for q, a in qa_pairs), 'Questions not aligned with answers'
        assert all(q['image_id'] == a['image_id'] for q, a in qa_pairs), 'Image id of question and answer don\'t match'
        assert questions['data_type'] == answers['data_type'], 'Mismatched data types'
        assert questions['data_subtype'] == answers['data_subtype'], 'Mismatched data subtypes'

    def _find_answerable(self, count=False):
        """ Create a list of indices into questions that will have at least one answer that is in the vocab """
        answerable = []
        if count:
            number_indices = torch.LongTensor([self.answer_to_index[str(i)] for i in range(0, 8)])
        for i, (answers, answer_len) in enumerate(self.answers):
            # store the indices of anything that is answerable
            if count:
                answers = answers[number_indices]
            answer_has_index = len(answers.nonzero()) > 0
            if answer_has_index:
                answerable.append(i)
        return answerable

    def _encode_question(self, question):
        """ Turn a question into a vector of indices and a question length """
        enc_q = [self.token_to_index['<start>']] + [self.token_to_index.get(word, self.token_to_index['<unk>']) for word in question] + [
            self.token_to_index['<end>']] + [self.token_to_index['<pad>']] * (self.max_question_length - len(question))
        # Find questions lengths
        q_len = len(question) + 2
        vec = torch.LongTensor(enc_q)
        return vec, torch.LongTensor([q_len])

    def _encode_answers(self, answers):
        """ Turn an answer into a vector """
        # answer vec will be a vector of answer counts to determine which answers will contribute to the loss.
        # this should be multiplied with 0.1 * negative log-likelihoods that a model produces and then summed up
        # to get the loss that is weighted by how many humans gave that answer
        enc_a = [self.answer_to_index.get(word, self.answer_to_index['<unk>']) for word in answers] + [self.answer_to_index['<pad>']] * (config.max_a_length - len(answers))
        # Find questions lengths
        a_len = len(answers)
        vec = torch.LongTensor(enc_a[:config.max_a_length])
        return vec, torch.LongTensor([a_len])

    def _load_image(self, image_id):
        """ Load an image """


        #image_id = str(image_id)
        #zero = ['0'] * (12 - len(image_id))
        #try:
        #    path = '/home/cike/VQG/train_images/' + image_id + '.jpg'
        #    img = Image.open(path).convert('RGB')
        #except:
        #    path = '/home/cike/VQG/train_images/' + image_id + '.jpg'
        #    img = Image.open(path).convert('RGB')
        #img = self.transform(img)

        # image_id = int(image_id)
        image_id = image_id

        #if image_id > 100000:
        #    image_id = image_id - 100000

        path = get_img(image_id)
        img = Image.open(path).convert('RGB')
        img = self.transform(img)

        return img

    def Image_adj_matrix(self, boxes, max_length):
        num_feat = len(boxes)
        relation_mask = np.zeros((max_length, max_length), dtype=np.float32)
        for i in range(num_feat):
            for j in range(i + 1, num_feat):
                if (boxes[i, 0] > boxes[j, 2] or boxes[j, 0] > boxes[i, 2]
                    or boxes[i, 1] > boxes[j, 3] or boxes[j, 1] > boxes[i, 3]):
                    pass
                else:
                    relation_mask[i, j] = relation_mask[j, i] = 1.0
        adj_A = relation_mask + np.eye(max_length, dtype=np.float32)
        # adj_A = np.ones((max_length, max_length))
        return adj_A

    def __getitem__(self, item):
        if self.answerable_only:
            item = self.answerable[item]
        q, q_length = self.questions[item]
        #difficult_level = torch.LongTensor([self.difficult_level[item]])
        # q_mask = torch.from_numpy((np.arange(self.max_question_length) < q_length).astype(int))
        a, a_length = self.answers[item]
        # a_mask = torch.from_numpy((np.arange(config.max_a_length) < a_length).astype(int))
        image_id = self.coco_ids[item]
        v = self._load_image(image_id)
        # if config.normalize_box:
        #     assert b.shape[1] == 4
        #     b[:, 0] = b[:, 0] / float(width)
        #     b[:, 1] = b[:, 1] / float(height)
        #     b[:, 2] = b[:, 2] / float(width)
        #     b[:, 3] = b[:, 3] / float(height)

        return v, q, a, item, q_length, image_id

    def __len__(self):
        if self.answerable_only:
            return len(self.answerable)
        else:
            return len(self.questions)


# this is used for normalizing questions
_special_chars = re.compile('[^a-z0-9 ]*')

# these try to emulate the original normalization scheme for answers
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))


def prepare_questions(questions_json):
    """ Tokenize and normalize questions from a given question json in the usual VQA format. """
    #questions = [q['question'] for q in questions_json['questions']]

    questions = [q['nlq'] for q in questions_json]

    for question in questions:
        question = question.lower()[:-1]
        question = _special_chars.sub('', question)
        yield question.split(' ')


def prepare_answers(answers_json):
    """ Normalize answers from a given answer json in the usual VQA format. """
    #answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in answers_json['annotations']]

    answers = [q['nla'] for q in answers_json]

    # print(answers)
    # The only normalization that is applied to both machine generated answers as well as
    # ground truth answers is replacing most punctuation with space (see [0] and [1]).
    # Since potential machine generated answers are just taken from most common answers, applying the other
    # normalizations is not needed, assuming that the human answers are already normalized.
    # [0]: http://visualqa.org/evaluation.html
    # [1]: https://github.com/VT-vision-lab/VQA/blob/3849b1eae04a0ffd83f56ad6f70ebd0767e09e0f/PythonEvaluationTools/vqaEvaluation/vqaEval.py#L96

    def process_punctuation(s):
        # the original is somewhat broken, so things that look odd here might just be to mimic that behaviour
        # this version should be faster since we use re instead of repeated operations on str's
        if _punctuation.search(s) is None:
            return s
        s = _punctuation_with_a_space.sub('', s)
        if re.search(_comma_strip, s) is not None:
            s = s.replace(',', '')
        s = _punctuation.sub(' ', s)
        s = _period_strip.sub('', s)
        return s.strip()

    for answer_list in answers:
        counter = Counter(answer_list)
        word, freq = counter.most_common(1)[0]
        if freq > 1:
            yield word.split()
        else:
            yield answer_list.split()


class CocoImages(data.Dataset):
    """ Dataset for MSCOCO images located in a folder on the filesystem """
    def __init__(self, path, transform=None):
        super(CocoImages, self).__init__()
        self.path = path
        self.id_to_filename = self._find_images()
        self.sorted_ids = sorted(self.id_to_filename.keys())  # used for deterministic iteration order
        print('found {} images in {}'.format(len(self), self.path))
        self.transform = transform

    def _find_images(self):
        id_to_filename = {}
        for filename in os.listdir(self.path):
            if not filename.endswith('.jpg'):
                continue
            id_and_extension = filename.split('_')[-1]
            id = int(id_and_extension.split('.')[0])
            id_to_filename[id] = filename
        return id_to_filename

    def __getitem__(self, item):
        id = self.sorted_ids[item]
        path = os.path.join(self.path, self.id_to_filename[id])
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return id, img

    def __len__(self):
        return len(self.sorted_ids)
