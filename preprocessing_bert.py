import os
import msgpack
from bert.tokenization_bert import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")


def convert_ids(line):
    tokens = line.strip().split("\t")

    doc_id, sent, label = tokens[0], tokens[1], int(tokens[2])
    # 0 : Negative, 1: Positive

    tokens = tokenizer.tokenize(sent)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    return token_ids, label


if __name__ == "__main__":
    train_data = open(os.path.join("data", "train.txt")).readlines()
    test_data = open(os.path.join("data", "test.txt")).readlines()

    train = [convert_ids(line) for line in train_data]
    test = [convert_ids(line) for line in test_data]

    data = {'train': train, 'test': test}

    with open('bert_data.msgpack', 'wb') as f:
        msgpack.dump(data, f, encoding='utf-8')
