import torch
from model import BERTDataset, BERTClassifier
import pandas as pd
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
import gluonnlp as nlp
import json


def classification(csv_data):
    with open('./config.json', 'r') as config_file:
        configs = json.load(config_file)

    checkpoint = configs["checkpoint"]
    save_path = configs["checkpoint_dir"]
    max_length = configs["max_length"]
    batch_size = configs["batch_size"]
    dr_rate = configs["dr_rate"]
    device = torch.device("cuda:0")
    dataframe = pd.read_csv(csv_data)

    mod, vocab = get_pytorch_kobert_model()
    bert_model = mod.to(device)
    # Set Tokenizer
    tok = get_tokenizer()
    tokenizer = nlp.data.BERTSPTokenizer(tok, vocab, lower=False)

    # Classifier init
    model = BERTClassifier(bert_model, dr_rate=dr_rate).to(device)

    ckpt = torch.load(checkpoint)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    sentiment_list = []
    with torch.no_grad():
        test_set = BERTDataset(dataframe['document'], 0, 1, tokenizer, max_length, True, False)
        test_input = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=4)

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_input):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            out = model(token_ids, valid_length, segment_ids)
            out = torch.sigmoid(out).cpu()

            for e in out:
                if e[0] >= 0.6:  # Negative
                    value = 'Negative'
                elif e[1] >= 0.6:  # Positive
                    value = 'Positive'
                else:  # Neutral
                    value = 'Neutral'
                sentiment_list.append(value)
    dataframe.assign(label=sentiment_list)
    dataframe.to_csv(save_path, index=False)


