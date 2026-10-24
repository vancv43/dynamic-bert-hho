# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     Ho Chi Minh City, Vietnam, vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Recompute the split statistics quoted in the manuscript from the saved CSV files."""
import os
import pandas as pd

D = os.path.join(os.path.dirname(__file__), "..", "data_processed")
tr, va, te = (pd.read_csv(os.path.join(D, f)).fillna("") for f in ("train.csv", "val.csv", "test.csv"))
print("sizes", len(tr), len(va), len(te), "pos rate", round(pd.concat([tr, va, te])["label"].mean(), 4))
if "user" in tr.columns:
    mtr = set(tr[tr.label == 1].user); mte = set(te[te.label == 1].user)
    print("malicious train users", len(mtr), "also in test", len(mtr & mte))
    for lab in (1, 0):
        c = tr[tr.label == lab].groupby("user").size()
        print("label", lab, "rows/user mean %.1f median %.1f max %d" % (c.mean(), c.median(), c.max()))
for name in ("train_grouped.csv", "val_grouped.csv", "test_grouped.csv", "train_loso.csv", "val_loso.csv", "test_loso.csv"):
    df = pd.read_csv(os.path.join(D, name))
    print(name, len(df), round(df.label.mean(), 4), df.user.nunique() if "user" in df else "")
g = {n: set(pd.read_csv(os.path.join(D, n)).user) for n in ("train_grouped.csv", "test_grouped.csv")}
print("grouped user overlap", len(g["train_grouped.csv"] & g["test_grouped.csv"]))
