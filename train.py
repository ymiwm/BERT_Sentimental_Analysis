import torch
from torch import nn
import msgpack
from bert.bert import Model
from optimization import BertAdam


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("bert_data.msgpack", "rb") as f:
        data = msgpack.load(f, encoding='utf-8')
    train, test = data['train'], data['test']

    model = Model()
    model.to(device)

    criterion = nn.BCELoss()

    num_train_epochs = 5
    batch_size = 32
    weight_decay = 0.0
    learning_rate = 1.5e-5  # 5e-5
    gradient_accumulation_steps = 1
    warmup_proportion = 0.1
    num_batch = (len(train) + batch_size - 1) // batch_size

    t_total = num_batch // gradient_accumulation_steps * num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate, warmup=warmup_proportion, t_total=t_total)

    for epoch in range(1, num_train_epochs + 1):
        num_data = len(train)
        num_batch = (num_data + batch_size - 1) // batch_size

        for ii in range(num_batch):
            start = ii * batch_size
            end = num_data if (ii + 1) * batch_size > num_data else (ii + 1) * batch_size

            batch_data = train[start:end]

            batch_bert_ids = [data[0] for data in batch_data]
            batch_labels_ids = [data[1] for data in batch_data]

            sent_lens = [len(words_ids) for words_ids in batch_bert_ids]
            max_sent_len = max(sent_lens)

            current_batch_size = len(batch_data)

            batch_berts = torch.LongTensor(current_batch_size, max_sent_len).fill_(0)

            for jj in range(current_batch_size):
                batch_berts[jj, :sent_lens[jj]] = torch.LongTensor(batch_bert_ids[jj])
            batch_labels = torch.FloatTensor(batch_labels_ids).unsqueeze(1)

            # to GPU devices
            batch_berts = batch_berts.to(device)
            batch_labels = batch_labels.to(device)

            output = model(batch_berts)

            loss = criterion(output, batch_labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if (ii + 1) % 1000 == 0:
                print("%6d/%d: loss %.6f" % (ii + 1, num_batch, loss.data))
                losses = []

        total = len(test)
        match = 0
        test_batch_size = 16

        num_data = len(test)
        num_batch = (num_data + test_batch_size - 1) // test_batch_size

        for ii in range(num_batch):
            start = ii * test_batch_size
            end = num_data if (ii + 1) * test_batch_size > num_data else (ii + 1) * test_batch_size

            batch_data = test[start:end]

            batch_bert_ids = [data[0] for data in batch_data]
            batch_labels_ids = [data[1] for data in batch_data]

            sent_lens = [len(words_ids) for words_ids in batch_bert_ids]
            max_sent_len = max(sent_lens)

            current_batch_size = len(batch_data)

            batch_berts = torch.LongTensor(current_batch_size, max_sent_len).fill_(0)

            for jj in range(current_batch_size):
                batch_berts[jj, :sent_lens[jj]] = torch.LongTensor(batch_bert_ids[jj])

            # to GPU devices
            batch_berts = batch_berts.to(device)

            batch_output = model(batch_berts)

            batch_output.data.cpu().numpy().tolist()

            batch_preds = [1 if output >= 0.5 else 0 for output in batch_output]

            for pred, answer in zip(batch_preds, batch_labels_ids):

                if pred == answer:
                    match += 1

        print("Epoch %d, ACC : %.2f" % (epoch, 100 * match / total))


if __name__ == "__main__":
    train()
