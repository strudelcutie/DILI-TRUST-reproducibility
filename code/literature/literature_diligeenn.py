                      







import pandas as pd, numpy as np, warnings, os, sys, time
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
NOW = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
ROOT = Path(__file__).resolve().parents[2]

FEAT = ROOT / "features" / "matrices"
TMP125 = ROOT / "results" / "tmp" / "stage12_5"
TAB125 = ROOT / "results" / "final" / "tables" / "stage12_5"
TAB9 = ROOT / "results" / "final" / "tables" / "stage9"
SDIR = TMP125 / "retraining" / "DILIGeNN"
for d in [SDIR, TAB125]:
    d.mkdir(parents=True, exist_ok=True)

TRAIN_SPLITS = ["scaffold", "temporal", "cluster_out"]

print("=" * 70)
print("STAGE 12.5 — DILIGeNN GraphSAGE REPRODUCTION")
print("=" * 70)

                 
import torch
import torch_geometric
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, matthews_corrcoef

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  PyTorch: {torch.__version__}, PyG: {torch_geometric.__version__}, Device: {device}")

                            
def mol_to_graph(smiles, label=0.0):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_features = []
    for atom in mol.GetAtoms():
        feats = [
            float(atom.GetAtomicNum()),
            float(atom.GetDegree()),
            float(atom.GetFormalCharge()),
            float(atom.GetTotalNumHs()),
            float(int(atom.GetHybridization())),
            float(int(atom.GetIsAromatic())),
            float(atom.GetMass()) / 100.0,
            float(atom.GetTotalValence()),
            float(atom.GetAtomicNum() == 6),
            float(atom.GetAtomicNum() == 7),
            float(atom.GetAtomicNum() == 8),
            float(atom.GetAtomicNum() == 9),
            float(atom.GetAtomicNum() == 15),
            float(atom.GetAtomicNum() == 16),
            float(atom.GetAtomicNum() == 17),
            float(atom.GetAtomicNum() == 35),
        ]
        atom_features.append(feats)

    x = torch.tensor(atom_features, dtype=torch.float)
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([[i, j], [j, i]])
    edge_index = torch.zeros((2, 0), dtype=torch.long) if not edges else torch.tensor(edges, dtype=torch.long).t().contiguous()
    y = torch.tensor([float(label)], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y)

               
class DILIGeNN_GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, hidden_channels=128, dropout_rate=0.2):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels * 2)
        self.ln1 = torch.nn.LayerNorm(hidden_channels * 2)
        self.conv2 = SAGEConv(hidden_channels * 2, hidden_channels)
        self.ln2 = torch.nn.LayerNorm(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels // 2)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.lin = torch.nn.Linear(hidden_channels // 2, 1)
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, getattr(data, 'batch', None)
        x = torch.relu(self.ln1(self.conv1(x, edge_index)))
        x = torch.relu(self.ln2(self.conv2(x, edge_index)))
        x = torch.relu(self.conv3(x, edge_index))
        x = self.dropout(x)
        if batch is not None:
            from torch_geometric.nn import global_mean_pool
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        return torch.sigmoid(self.lin(x)).squeeze(-1)

                  
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, n = 0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch), batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(1, n)

def predict_loader(model, loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            scores.extend(model(batch).cpu().numpy().tolist())
    return np.array(scores)

                        
print("\n--- Loading SMILES ---")
smiles_map = {}

                                                      
for f in sorted(ROOT.rglob("features/**/*.csv")):
    try:
        d = pd.read_csv(f, nrows=3)
        smi_col = next((c for c in d.columns if c.lower() in ("smiles","canonical_smiles","rdkit_smiles")), None)
        id_col = next((c for c in d.columns if c.lower() == "sample_id"), None)
        if smi_col and id_col:
            d_full = pd.read_csv(f)
            for _, r in d_full.iterrows():
                sid = str(r[id_col])
                if sid not in smiles_map:
                    smiles_map[sid] = str(r[smi_col])
    except:
        pass

                                                                  
cm_f = ROOT / "data_curated" / "compounds" / "compound_master.csv"
if cm_f.exists():
    cm = pd.read_csv(cm_f)
    if "canonical_smiles" in cm.columns:
        for _, r in cm.iterrows():
            cid = str(r.get("compound_id",""))
            if cid not in smiles_map:
                smiles_map[cid] = str(r["canonical_smiles"])

                                           
all_sids = set()
for sn in TRAIN_SPLITS:
    for role in ["train", "valid", "test"]:
        fname = f"model_ready_{sn}_{'temporal_test' if sn == 'temporal' and role == 'test' else role}.csv"
        fp = FEAT / sn / fname
        if fp.exists():
            all_sids.update(pd.read_csv(fp)["sample_id"].astype(str).tolist())
for sn, sd in [("external_dilist","external_dilist"), ("post2021_lockbox","post2021_lockbox")]:
    fp = FEAT / sd / f"model_ready_{sn}_scaffold_scaler.csv"
    if fp.exists():
        all_sids.update(pd.read_csv(fp)["sample_id"].astype(str).tolist())

print(f"  SMILES available: {len(smiles_map)}, unique sample IDs: {len(all_sids)}")
print(f"  Overlap: {len(set(smiles_map.keys()) & all_sids)} samples with SMILES")

                                               
print("  Building graphs...")
graphs = {}
for sid in all_sids:
    smi = smiles_map.get(sid)
    if smi:
        g = mol_to_graph(smi, label=0.0)
        if g is not None:
            graphs[sid] = g

print(f"  Graphs built: {len(graphs)}/{len(all_sids)}")

                          
print("\n--- Training & Evaluation ---")
diligeenn_metrics = []
diligeenn_log = []

for model_split in TRAIN_SPLITS:
    train_f = FEAT / model_split / f"model_ready_{model_split}_train.csv"
    valid_f = FEAT / model_split / f"model_ready_{model_split}_valid.csv"
    if not train_f.exists(): continue

    tr = pd.read_csv(train_f)
    vl = pd.read_csv(valid_f)
    tr_sids = tr["sample_id"].astype(str).tolist()
    vl_sids = vl["sample_id"].astype(str).tolist()
    yt = tr["label"].values.astype(int)
    yv = vl["label"].values.astype(int)

                                   
    train_g, valid_g = [], []
    for sid, lbl in zip(tr_sids, yt):
        g = graphs.get(sid)
        if g is not None:
            g = g.clone()
            g.y = torch.tensor([float(lbl)], dtype=torch.float)
            train_g.append(g)
    for sid, lbl in zip(vl_sids, yv):
        g = graphs.get(sid)
        if g is not None:
            g = g.clone()
            g.y = torch.tensor([float(lbl)], dtype=torch.float)
            valid_g.append(g)

    print(f"\n  {model_split}: {len(train_g)} train, {len(valid_g)} valid graphs")

    if len(train_g) < 20:
        diligeenn_log.append({"split": model_split, "status": "skipped", "n_graphs": len(train_g)})
        continue

    num_features = train_g[0].x.shape[1]
    model = DILIGeNN_GraphSAGE(num_features, hidden_channels=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.BCELoss()
    train_loader = DataLoader(train_g, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_g, batch_size=64, shuffle=False)

    for epoch in range(80):
        train_epoch(model, train_loader, optimizer, criterion)

    yv_pred = predict_loader(model, valid_loader)
    auroc_v = roc_auc_score(yv[:len(yv_pred)], yv_pred[:len(yv)])
    print(f"    Valid AUROC: {auroc_v:.4f}")

                            
    for eval_split, test_dir, test_fname in [
        ("scaffold", "scaffold", "model_ready_scaffold_test.csv"),
        ("temporal", "temporal", "model_ready_temporal_temporal_test.csv"),
        ("cluster_out", "cluster_out", "model_ready_cluster_out_test.csv"),
        ("external_dilist", "external_dilist", "model_ready_external_dilist_scaffold_scaler.csv"),
        ("post2021_lockbox", "post2021_lockbox", "model_ready_post2021_lockbox_scaffold_scaler.csv"),
    ]:
        fp = FEAT / test_dir / test_fname
        if not fp.exists(): continue
        src = "scaffold" if eval_split in ("external_dilist","post2021_lockbox") else eval_split
        if src != model_split: continue

        d_test = pd.read_csv(fp)
        test_sids = d_test["sample_id"].astype(str).tolist()
        yt_test = d_test["label"].values if "label" in d_test.columns else np.full(len(d_test), np.nan)

        test_g_list = []
        valid_indices = []
        for i, sid in enumerate(test_sids):
            g = graphs.get(sid)
            if g is not None:
                g = g.clone()
                g.y = torch.tensor([0.0], dtype=torch.float)
                test_g_list.append(g)
                valid_indices.append(i)

        if len(test_g_list) == 0:
            y_score = np.full(len(d_test), 0.5)
        else:
            test_loader = DataLoader(test_g_list, batch_size=64, shuffle=False)
            y_score_partial = predict_loader(model, test_loader)
            y_score = np.full(len(d_test), 0.5)
            for j, idx in enumerate(valid_indices):
                y_score[idx] = y_score_partial[j]
        y_score = np.clip(y_score, 0, 1)

        df_pred = pd.DataFrame({"sample_id": d_test["sample_id"].values, "split": eval_split,
            "model_name": "DILIGeNN_GraphSAGE", "seed": "single",
            "y_true": yt_test, "raw_score": np.round(y_score, 6),
            "fold_role": "test", "reproduction_class": "A_adapted",
            "graph_available": np.array([i in valid_indices for i in range(len(d_test))])})
        df_pred.to_csv(SDIR / f"{eval_split}_raw_predictions.csv", index=False)

        if np.all(np.isnan(yt_test)): continue
        has_both = len(set(yt_test)) > 1
        yp = (y_score >= 0.5).astype(int)
        diligeenn_metrics.append({"model_name": "DILIGeNN_GraphSAGE", "split": eval_split,
            "reproduction_class": "A", "metric_source": "actual_prediction",
            "n": len(yt_test), "n_with_graphs": len(test_g_list),
            "auroc": round(roc_auc_score(yt_test, y_score), 4) if has_both else "NA",
            "auprc": round(average_precision_score(yt_test, y_score), 4) if has_both else "NA",
            "brier": round(brier_score_loss(yt_test, y_score), 4),
            "mcc": round(matthews_corrcoef(yt_test, yp), 4)})
        print(f"    {eval_split}: AUROC={diligeenn_metrics[-1]['auroc']} ({len(test_g_list)}/{len(d_test)} graphs) Brier={diligeenn_metrics[-1]['brier']}")

    diligeenn_log.append({"split": model_split, "status": "completed", "valid_auroc": round(auroc_v, 4),
                          "n_train_graphs": len(train_g), "n_valid_graphs": len(valid_g)})

                      
pd.DataFrame(diligeenn_log).to_csv(SDIR / "training_log.csv", index=False)
df_dgnn = pd.DataFrame(diligeenn_metrics)
if len(df_dgnn) > 0:
    df_dgnn.to_csv(TAB125 / "literature_model_diligeenn_metrics.csv", index=False)
    print(f"\n  Results saved: {len(df_dgnn)} metric rows")

    s9 = pd.read_csv(TAB9 / "stage9_all_split_metrics.csv")
    hgb = s9[s9["model"] == "HistGB"]
    print(f"\n  DILIGeNN vs HistGB:")
    for _, r in df_dgnn.iterrows():
        hgb_row = hgb[hgb["split"] == r["split"]]
        hgb_auc = hgb_row.iloc[0]["auroc"] if len(hgb_row) > 0 else None
        if hgb_auc and r["auroc"] != "NA":
            delta = round(float(r["auroc"]) - float(hgb_auc), 4)
            print(f"    {r['split']}: DILIGeNN={r['auroc']} HistGB={hgb_auc} delta={delta:+}")

print(f"\n  DILIGeNN reproduction complete. Graphs: {len(graphs)}/{len(all_sids)} samples")
