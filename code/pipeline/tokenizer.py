import collections
import unicodedata

def _is_punctuation(char):
    cp = ord(char)
    if((cp>=33 and cp<=47) or (cp>=58 and cp<=64) or(cp>=91 and cp<=96) or (cp>=123 and cp<=126)):
        return True

    cat = unicodedata.category(char)
    if cat.startswith('P'):
        return True
    return False
    
def _is_control(char):
    if char == '\t' or char == '\n' or char == '\r':
        return False
    cat = unicodedata.category(char)
    if cat.startswith('C'):
        return True
    return False

def _is_whitespace(char):
    if char == ' ' or char == '\t' or char == '\n' or char == '\r':
        return True
    cat = unicodedata.category(char)
    if cat == 'Zs':
        return True
    return False

def _run_strip_accents(text):
    text = unicodedata.normalize('NFD', text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat =='Mn':
            continue
        output.append(char)
    return ''.join(output)

def _run_split_on_punc(text):
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if _is_punctuation(char):
            output.append([char])
            start_new_word =True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1
    return [''.join(x) for x in output]

def whitespace_tokenize(text):
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def convert2unicode(text):
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode('utf-8', 'ignore')
    else:
        raise ValueError('Unsupported string type: %s' % (type(text)))

def tokenize(text):
    org_tokens = whitespace_tokenize(text)
    split_tokens = []
    for token in org_tokens:
        token = token.lower()
        token = _run_strip_accents(token)
        split_tokens.extend(_run_split_on_punc(token))
    return whitespace_tokenize(' '.join(split_tokens))

def remove_stopwords(text):
    #text: list
    output_tokens = []

    for i in text:
        _ = list(map(_is_punctuation, i))
        word = []
        for w, p in zip(i, _):
            if p == False:
                word.append(w)
                output_tokens.append(''.join(word))

    return list(filter(None, output_tokens))

def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    index = 0
    reader = open(vocab_file).readlines()
    for item in reader:
        token = convert2unicode(item)
        token = token.strip()
        vocab[token] = index
        index += 1
    return vocab

def convert_by_vocab(vocab, items):
    output = []
    for item in items:
        output.append(vocab[item])
    return output

def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab_tokens)

def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)

####BERT Tokenizer
class FullTokenizer(object):
    def __init__(self, vocab_file, do_lower_case=True):
        #vocab_file {word:index}
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v:k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        #text str -> token list
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)

class BasicTokenizer(object):
    def __init__(self, do_lower_case):
        self.do_lower_case = do_lower_case
    
    def tokenize(self, text):
        text = convert2unicode(text)
        text = self._clean_text(text)

        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                #remove stress symbol
                token = self._run_strip_accents(token)

            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(' '.join(split_tokens) )

        return output_tokens

    def _run_strip_accents(self, text):
        text = unicodedata.normalize('NFD', text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == 'Mn':
                continue
            output.append(char)
        return ''.join(output)

    def _run_split_on_punc(self, text):
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1 

        return [''.join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        # add white space
        output = []

        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(' ')
                output.append(char)
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

    def _is_chinese_char(self, cp):
        # check CJK
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or (cp >= 0x3400 and cp <= 0x4DBF) or (cp >= 0x20000 and cp <= 0x2A6DF) or (cp >= 0x2A700 and cp <= 0x2B73F) or (cp >= 0x2B740 and cp <= 0x2B81F) or (cp >= 0x2B820 and cp <= 0x2CEAF) or (cp >= 0xF900 and cp <= 0xFAFF) or (cp >= 0x2F800 and cp <= 0x2FA1F)):
            return True
        return False

    def _clean_text(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

class WordpieceTokenizer(object):
    def __init__(self, vocab, unk_token='[UNK]', max_input_chars_per_word=200 ):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        text = convert2unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
            is_bad = False
            start = 0
            sub_tokens = []

            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = ''.join(chars[start:end])
                    if start > 0:
                        substr = '##' + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
                
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
'''
tokenizer = FullTokenizer('/home/joli/bert_base/vocab.txt')
a = ['[CLS]']+ (tokenizer.tokenize('I have been running')) + ['[SEP]']
b = (tokenizer.tokenize('習近平小熊維尼'))

print(a)
print(tokenizer.convert_tokens_to_ids(a))
'''
