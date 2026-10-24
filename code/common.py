# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     Ho Chi Minh City, Vietnam, vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Shared BERT dataset/train/eval utilities for the Dynamic BERT-HHO experiment."""
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, cohen_kappa_score
)
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification, BertTokenizerFast

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
MODEL_NAME = "bert-base-uncased"

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    return _tokenizer


class HttpTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=96):
        enc = tokenizer(
            list(texts), truncation=True, padding="max_length", max_length=max_len,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def build_model(dropout, trainable_layers):
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
        attn_implementation="eager",  # eager attention: same computation, supports attention dropout on every backend
    )
    # freeze all encoder layers except the last `trainable_layers`
    total_layers = len(model.bert.encoder.layer)
    n_freeze = max(0, total_layers - int(round(trainable_layers)))
    for p in model.bert.embeddings.parameters():
        p.requires_grad = False
    for i, layer in enumerate(model.bert.encoder.layer):
        req_grad = i >= n_freeze
        for p in layer.parameters():
            p.requires_grad = req_grad
    return model.to(DEVICE)


def train_and_eval(train_texts, train_labels, eval_texts, eval_labels, hp, verbose=False):
    """hp: dict with lr, batch_size, dropout, weight_decay, epochs, trainable_layers"""
    tokenizer = get_tokenizer()
    train_ds = HttpTextDataset(train_texts, train_labels, tokenizer)
    eval_ds = HttpTextDataset(eval_texts, eval_labels, tokenizer)

    bs = int(round(hp["batch_size"]))
    model = build_model(hp["dropout"], hp["trainable_layers"])
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=hp["lr"], weight_decay=hp["weight_decay"],
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True)
    model.train()
    epochs = max(1, int(round(hp["epochs"])))
    for ep in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            optim.step()
            total_loss += out.loss.item()
        if verbose:
            print(f"  epoch {ep+1}/{epochs} loss={total_loss/len(train_loader):.4f}")

    metrics, probs, preds = evaluate(model, eval_ds)
    return model, metrics, probs, preds


@torch.no_grad()
def evaluate(model, eval_ds, batch_size=32):
    model.eval()
    loader = torch.utils.data.DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
    all_probs, all_preds, all_labels = [], [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn = batch["attention_mask"].to(DEVICE)
        out = model(input_ids=input_ids, attention_mask=attn)
        probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(batch["labels"].numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")
    kappa = cohen_kappa_score(all_labels, all_preds)
    metrics = {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc, "kappa": kappa,
    }
    return metrics, all_probs, all_preds
