"""Dynamic BERT-HHO-Meta: BERT contextual embedding fused with structural/behavioural
metadata features (content length, digit ratio, URL structure, time-of-day) via a small
fusion head. Architecturally distinct from every baseline in the paper's related-work
comparison table (Table 2), which use either pure text features or pure structured
features, never a joint fusion tuned end-to-end by HHO."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, cohen_kappa_score
)
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizerFast

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_NAME = "bert-base-uncased"

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    return _tokenizer


class HttpMetaDataset(Dataset):
    def __init__(self, texts, meta, labels, tokenizer, max_len=96):
        enc = tokenizer(
            list(texts), truncation=True, padding="max_length", max_length=max_len,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.meta = torch.tensor(np.asarray(meta, dtype=np.float32))
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "meta": self.meta[idx],
            "labels": self.labels[idx],
        }


class BertMetaFusion(nn.Module):
    """BERT [CLS] embedding fused with a structural/behavioural metadata vector through a
    shallow fusion head. `fusion_hidden` and `fusion_dropout` are the two additional
    hyperparameters HHO searches over on top of the original six BERT hyperparameters."""

    def __init__(self, dropout, meta_dim, fusion_hidden=32, fusion_dropout=0.2, text_proj_dim=64):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME, hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
                                              attn_implementation="eager")
        hidden = self.bert.config.hidden_size
        text_proj_dim = int(max(8, round(text_proj_dim)))
        fusion_hidden = int(max(4, round(fusion_hidden)))
        self.text_proj = nn.Sequential(nn.Linear(hidden, text_proj_dim), nn.ReLU())
        self.meta_proj = nn.Sequential(nn.Linear(meta_dim, fusion_hidden), nn.ReLU())
        self.fusion_dropout = nn.Dropout(fusion_dropout)
        self.classifier = nn.Linear(text_proj_dim + fusion_hidden, 2)

    def forward(self, input_ids, attention_mask, meta):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        text_h = self.text_proj(cls)
        meta_h = self.meta_proj(meta)
        fused = self.fusion_dropout(torch.cat([text_h, meta_h], dim=-1))
        return self.classifier(fused)


def build_model(dropout, trainable_layers, meta_dim, fusion_hidden=32, fusion_dropout=0.2):
    model = BertMetaFusion(dropout, meta_dim, fusion_hidden, fusion_dropout)
    total_layers = len(model.bert.encoder.layer)
    n_freeze = max(0, total_layers - int(round(trainable_layers)))
    for p in model.bert.embeddings.parameters():
        p.requires_grad = False
    for i, layer in enumerate(model.bert.encoder.layer):
        req_grad = i >= n_freeze
        for p in layer.parameters():
            p.requires_grad = req_grad
    return model.to(DEVICE)


def train_and_eval_meta(train_texts, train_meta, train_labels, eval_texts, eval_meta, eval_labels, hp, verbose=False):
    """hp: dict with lr, batch_size, dropout, weight_decay, epochs, trainable_layers,
    fusion_hidden, fusion_dropout"""
    tokenizer = get_tokenizer()
    meta_dim = np.asarray(train_meta).shape[1]
    train_ds = HttpMetaDataset(train_texts, train_meta, train_labels, tokenizer)
    eval_ds = HttpMetaDataset(eval_texts, eval_meta, eval_labels, tokenizer)

    bs = int(round(hp["batch_size"]))
    model = build_model(hp["dropout"], hp["trainable_layers"], meta_dim, hp.get("fusion_hidden", 32), hp.get("fusion_dropout", 0.2))
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=hp["lr"], weight_decay=hp["weight_decay"],
    )
    loss_fn = nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True)
    model.train()
    epochs = max(1, int(round(hp["epochs"])))
    for ep in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            meta = batch["meta"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            logits = model(input_ids, attn, meta)
            loss = loss_fn(logits, labels)
            loss.backward()
            optim.step()
            total_loss += loss.item()
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
        meta = batch["meta"].to(DEVICE)
        logits = model(input_ids, attn, meta)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
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
