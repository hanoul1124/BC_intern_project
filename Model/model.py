import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm.auto import tqdm
import json
import datetime

# kobert settings
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer

# transformers settings
from transformers.optimization import get_cosine_schedule_with_warmup

# import torch_xla
# import torch_xla.core.xla_model as xm


# Dataset to Token(as input data for BERT model)
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


# Classifier Modeling
class BERTClassifier(nn.Module):
    def __init__(
        self,
        bert,  # bert model
        hidden_size=768,  # hidden node size(fc layer)
        num_classes=2,  # # of target class
        dr_rate=None,  # dropout rate
        params=None
    ):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(
            input_ids=token_ids,
            token_type_ids=segment_ids.long(),
            attention_mask=attention_mask.float().to(token_ids.device)
        )

        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler

        return self.classifier(out)


# define function : caculate accuracy
def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
    return acc


def save_model(project_name, model, optimizer):
    now = datetime.datetime.now()
    time_str = now.strftime("%y%m%d%H%M")
    MODEL_NAME = f'{project_name}_{time_str}.pt'
    MODEL_SAVE_PATH = os.path.join("./checkpoint/")

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"{MODEL_SAVE_PATH} -- Directory already exists \n")
    else:
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        print(f"{MODEL_SAVE_PATH} -- Directory created \n")

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        MODEL_SAVE_PATH + MODEL_NAME
    )
    print(f"Model saved: {MODEL_SAVE_PATH + MODEL_NAME}")


def training(train_set, test_set, checkpoint=None):
    # Set parameters
    with open('./config.json', 'r') as config_file:
        configs = json.load(config_file)
    max_len = configs["max_length"]  # The number of tokens of sentence made by Tokenizer
    batch_size = configs["batch_size"]
    warmup_ratio = configs["warmup_ratio"]
    num_epochs = configs["num_epochs"]
    max_grad_norm = configs["max_grad_norm"]
    log_interval = configs["log_interval"]
    learning_rate = configs["learning_rate"]
    dr_rate = configs["dr_rate"]
    num_cpu = os.cpu_count()

    # Set Device
    # device = xm.xla_device()  # TPU
    device = torch.device("cuda:0")  # GPU

    # Set BERT Model
    mod, vocab = get_pytorch_kobert_model()
    bert_model = mod.to(device)
    # Set Tokenizer
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    # Classifier init
    model = BERTClassifier(bert_model, dr_rate=dr_rate).to(device)

    # Train from (pretrained) model checkpoint
    if checkpoint:
        ckpt = torch.load(checkpoint)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)

    dataset_train = nlp.data.TSVDataset(
        train_set,
        field_indices=[1, 2],
        num_discard_samples=1
    )
    dataset_test = nlp.data.TSVDataset(
        test_set,
        field_indices=[1, 2],
        num_discard_samples=1
    )

    # BERTDataset
    data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
    data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

    # dataloader
    train_dataloader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, num_workers=num_cpu)
    test_dataloader = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size, num_workers=num_cpu)

    # Optimizer/Loss function Settings
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Setting scheduler(Learning rate decay)
    total_training_steps = len(train_dataloader) * num_epochs
    warmup_step = int(total_training_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_step,
        num_training_steps=total_training_steps
    )

    # Train/Test BERT Classifier
    for epoch in range(num_epochs):
        train_acc = 0.0
        test_acc = 0.0

        model.train()  # Set Training session
        for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(train_dataloader),
                                                                            total=len(train_dataloader)):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(device)  # 각 토큰의 vocab dict 번호
            segment_ids = segment_ids.long().to(device)  # 0번째 문장의 토큰
            valid_length = valid_length  # 유효 토큰의 수(패딩 제외)
            label = label.long().to(device)
            # forward pass
            out = model(token_ids, valid_length, segment_ids)
            # loss
            loss = loss_fn(out, label)
            # back prop
            loss.backward()
            # update parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            # xm.optimizer_step(optimizer, barrier=True)  # TPU option
            optimizer.step()
            scheduler.step()

            # accumulate train_acc
            train_acc += calc_accuracy(out, label)  # we'll calculate mean(accumulated_train_acc)

            if batch_id % log_interval == 0:  # print every 200 batch
                print(
                    f"epoch {epoch + 1} batch id {batch_id + 1} loss {loss.data.cpu().numpy()} train acc {train_acc / (batch_id + 1)}"
                )
        print(f"epoch {epoch + 1} train acc {train_acc / (batch_id + 1)}")

        model.eval()  # Set Test session
        for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(test_dataloader),
                                                                            total=len(test_dataloader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)

            out = model(token_ids, valid_length, segment_ids)  # Make Prediction
            test_acc += calc_accuracy(out, label)

        print(f"epoch {epoch + 1} test acc {test_acc / (batch_id + 1)}")
        save_model(
            project_name="SentmentClassifier",
            model=model,
            optimizer=optimizer
        )

