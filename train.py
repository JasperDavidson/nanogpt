from helpers import Tokenizer, extract_input, extract_vocab


def extract_data():
    input = extract_input()
    vocab = extract_vocab()
    t = Tokenizer()
    t.generate_tokenizer(vocab)

    data = t.get_data_split(input, val_percentage=0.1, test_percentage=0.1)
    print(data.train[:1000])
