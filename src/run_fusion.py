"""ONE-STEP fusion experiment: combines your trained CNN and MLP and evaluates everything.

What you do:
  1. Set CNN_RUN_DIR and MLP_RUN_DIR below (the run folder, or the model_weights.pt inside it).
  2. Run:  python run_fusion.py

What it does, automatically:
  - Extracts each model's features + logits for train and val (extract_outputs.py) the first time,
    and reuses the saved files afterwards (re-extracts if you point it at different runs).
  - Builds three kinds of fused model on top of those saved numbers (no images re-processed):
        stack   a tiny logistic regression on the two models' final scores (logits)
        linear  logistic regression on the CNN features + MLP features joined together
        mlp     a small neural net (one hidden layer) on the joined features
    Each is trained two ways:
        "fit on train"  learns on the train split's features, tested on val
        "CV on val"     learns on val itself with 5-fold cross-validation (grouped by lesion, so a
                        lesion is never on both sides). This is the safer check, because the CNN has
                        already seen the train images and its train-set outputs look better than they
                        will on new images, which can make a "fit on train" head over-trust the CNN.
  - Compares everything against the CNN alone and the MLP alone on the same val images.
  - Writes a timestamped folder under results/ with:
        comparison.csv / comparison.txt   the headline table (AUC with CI, recall at fixed specificity...)
        matched_specificity_by_diagnosis.csv   which diagnoses each model flags at the same false-alarm rate
        one sub-folder per model with: confusion_matrix.csv, classification_report.csv,
            per_diagnosis_val.csv, per_image_predictions_val.csv, threshold_by_diagnosis.csv,
            threshold_overall.csv, plus the heatmap / threshold-sweep / strip plots
            (the same analysis multiclass_analysis.py produces)
        fused_models/*.joblib             the heads trained on the train split
        run_summary.json                  which runs / settings produced this
  Only the first run needs the GPU/torch (extraction); everything after that is fast scikit-learn.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Settings — only the first two lines need editing
# ---------------------------------------------------------------------------
CNN_RUN_DIR = Path("results/2026-10-29_0124_M-EfficientNetB2_Tune_binary_S-size-25331_E-50")   # folder (or .pt file) of the trained image model
MLP_RUN_DIR = Path("results/2026-10-23_0038_M-ComplexMLP_binary_S-size-25331_E-50")   # folder (or .pt file) of the trained metadata model

TRAIN_METADATA_CSV = "../data/processed/train_metadata.csv"
VAL_METADATA_CSV = "../data/processed/val_metadata.csv"
FEATURES_DIR = Path("fusion_features")      # where features_train.npz / features_val.npz are kept
FORCE_REEXTRACT = False                     # True = redo extraction even if the files look current
RESULTS_ROOT = Path("results")

HEADS = ["stack", "linear", "mlp"]          # which fused models to build
CUTOFF = 0.5                                # cutoff for the confusion matrices / classification reports
SPEC_LEVELS = (0.80, 0.85, 0.90)            # "malignant recall when benign recall is X"
MATCHED_SPEC = 0.85                         # specificity used for the per-diagnosis comparison
N_BOOT = 500                                # bootstrap resamples (by lesion) for the confidence intervals
PRIMARY = "auto"                            # fused model to give the image grid; "auto" = best val AUC among fused
MAKE_IMAGE_GRID = True                      # the MEL/BKL/NV example-image grid for the primary model
SEED = 42

HEAD_LABEL = {"stack": "Stack (logits)", "linear": "Linear (features)", "mlp": "MLP head (features)"}


# ---------------------------------------------------------------------------
# Step 1: make sure the feature files exist and belong to these two runs
# ---------------------------------------------------------------------------
def run_dir_of(path):
    path = Path(path)
    return path.parent if path.suffix == ".pt" else path


def features_current(npz_path, cnn_dir, mlp_dir):
    if not npz_path.exists():
        return False
    try:
        info = json.loads(str(np.load(npz_path)["info_json"]))
    except Exception:
        return False
    return info.get("cnn_run") == str(cnn_dir.resolve()) and info.get("mlp_run") == str(mlp_dir.resolve())


def ensure_features():
    cnn_dir, mlp_dir = run_dir_of(CNN_RUN_DIR), run_dir_of(MLP_RUN_DIR)
    for label, d in [("CNN", cnn_dir), ("MLP", mlp_dir)]:
        if not (d / "model_weights.pt").exists():
            raise FileNotFoundError(f"No model_weights.pt in {d}. Set {label}_RUN_DIR at the top of run_fusion.py.")
    paths = {s: FEATURES_DIR / f"features_{s}.npz" for s in ("train", "val")}
    missing = [s for s, p in paths.items() if FORCE_REEXTRACT or not features_current(p, cnn_dir, mlp_dir)]
    if missing:
        print(f"Extracting features for: {', '.join(missing)} (one-time; this runs the CNN over the images)")
        from extract_outputs import run_extraction   # imported here so a re-run with saved files needs no torch
        run_extraction(CNN_RUN_DIR, MLP_RUN_DIR, missing, FEATURES_DIR, TRAIN_METADATA_CSV, VAL_METADATA_CSV)
    else:
        print(f"Using saved features in {FEATURES_DIR}/ (made from these exact runs).")
    return paths


def load_features(path):
    z = np.load(path, allow_pickle=True)   # our own file; older ones stored the name columns as objects
    data = {k: z[k] for k in z.files if k != "info_json"}
    for k in ("image", "diagnosis", "lesion_id_filled"):
        data[k] = data[k].astype(str)
    return data, json.loads(str(z["info_json"]))


# ---------------------------------------------------------------------------
# Step 2: the fused models
# ---------------------------------------------------------------------------
def margin(logits):
    """Malignant score minus benign score: one number per image (log-odds of malignant)."""
    return logits[:, 1] - logits[:, 0]


def inputs_for(head, d):
    if head == "stack":
        return np.c_[margin(d["cnn_logits"]), margin(d["mlp_logits"])]
    return np.c_[d["cnn_features"], d["mlp_features"]]


def build_head(head):
    if head == "stack":
        return LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    if head == "linear":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.01, class_weight="balanced", max_iter=3000))
    if head == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(256,), alpha=1e-2, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=10, max_iter=200, random_state=SEED))
    raise ValueError(head)


def fit_head(head, X, y):
    model = build_head(head)
    if head == "mlp":   # sklearn's MLP has no class weights, so repeat malignant rows until classes are balanced
        rng = np.random.default_rng(SEED)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        minority, majority = (pos, neg) if len(pos) < len(neg) else (neg, pos)
        extra = rng.choice(minority, len(majority) - len(minority), replace=True)
        idx = rng.permutation(np.concatenate([np.arange(len(y)), extra]))
        X, y = X[idx], y[idx]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    return model


def p_malignant(model, X):
    return model.predict_proba(X)[:, 1]


def cv_on_val(head, X, y, groups):
    """Out-of-fold P(malignant) for every val image; folds never split a lesion."""
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups):
        oof[te] = p_malignant(fit_head(head, X[tr], y[tr]), X[te])
    return oof


def softmax_malignant(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


# ---------------------------------------------------------------------------
# Step 3: metrics
# ---------------------------------------------------------------------------
def recall_at_spec(y, p, spec):
    fpr, tpr, _ = roc_curve(y, p)
    return float(tpr[np.searchsorted(fpr, 1 - spec, side="right") - 1])


def threshold_at_spec(y, p, spec):
    fpr, _, thr = roc_curve(y, p)
    return thr[np.searchsorted(fpr, 1 - spec, side="right") - 1]


def headline(y, p, cutoff=CUTOFF):
    pred = (p >= cutoff).astype(int)
    tp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum())
    tn, fp = int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum())
    mal_rec, ben_rec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f2 = 5 * prec * mal_rec / (4 * prec + mal_rec) if (4 * prec + mal_rec) > 0 else 0.0
    row = {"AUC": roc_auc_score(y, p), "accuracy": (tp + tn) / len(y), "balanced_acc": (mal_rec + ben_rec) / 2,
           "malignant_recall": mal_rec, "benign_recall": ben_rec, "F2": f2}
    for s in SPEC_LEVELS:
        row[f"mal_recall@spec{s:.2f}"] = recall_at_spec(y, p, s)
    return row


def lesion_bootstrap_indices(groups, n_boot, seed=0):
    codes, uniq = pd.factorize(pd.Series(groups))
    members = [np.where(codes == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    return [np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))]) for _ in range(n_boot)]


def bootstrap_auc(y, scores, boots, ref=None):
    """95% CI of AUC (and of AUC minus the reference model's AUC, on the same resamples)."""
    aucs, diffs = [], []
    for idx in boots:
        if y[idx].min() == y[idx].max():
            continue
        a = roc_auc_score(y[idx], scores[idx])
        aucs.append(a)
        if ref is not None:
            diffs.append(a - roc_auc_score(y[idx], ref[idx]))
    out = {"AUC_lo": np.percentile(aucs, 2.5), "AUC_hi": np.percentile(aucs, 97.5)}
    if ref is not None:
        out.update({"dAUC_vs_CNN": np.mean(diffs), "dAUC_lo": np.percentile(diffs, 2.5),
                    "dAUC_hi": np.percentile(diffs, 97.5)})
    return out


# ---------------------------------------------------------------------------
# Step 4: per-model outputs (same analysis multiclass_analysis.py produces)
# ---------------------------------------------------------------------------
def slug(name):
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def analyze_model(name, p, df, y, out_dir):
    from multiclass_analysis import (summarize_by_diagnosis, save_per_image_predictions, plot_summary,
                                     threshold_sweep_by_diagnosis, overall_threshold_metrics,
                                     plot_threshold_sweep, plot_stripplot)
    from config import BINARY_CLASSES
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = (p >= CUTOFF).astype(int)

    pred_path = out_dir / "per_image_predictions_val.csv"
    save_per_image_predictions(df, preds, p, pred_path)

    summary = summarize_by_diagnosis(df, preds, p)
    summary.to_csv(out_dir / "per_diagnosis_val.csv", index=False)

    cm = confusion_matrix(y, preds, labels=[0, 1])
    pd.DataFrame(cm, index=[f"true_{c}" for c in BINARY_CLASSES],
                 columns=[f"pred_{c}" for c in BINARY_CLASSES]).to_csv(out_dir / "confusion_matrix.csv")
    pd.DataFrame(classification_report(y, preds, labels=[0, 1], target_names=BINARY_CLASSES, output_dict=True,
                                       zero_division=0)).T.to_csv(out_dir / "classification_report.csv")

    by_dx, overall = threshold_sweep_by_diagnosis(df, p), overall_threshold_metrics(y, p)
    by_dx.to_csv(out_dir / "threshold_by_diagnosis.csv", index=False)
    overall.to_csv(out_dir / "threshold_overall.csv", index=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)   # seaborn's palette/hue notice in plot_stripplot
        plot_summary(summary, f"{name}: val predictions by diagnosis (cutoff {CUTOFF:.2f})",
                     out_dir / "summary_heatmap.png")
        plot_threshold_sweep(by_dx, overall, name, out_dir / "threshold_sweep.png")
        plot_stripplot(pred_path, out_dir / "stripplot.png")
    return pred_path, summary, cm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    np.random.seed(SEED)
    paths = ensure_features()
    tr, info = load_features(paths["train"])
    va, info_va = load_features(paths["val"])
    if info_va.get("cnn_run") != info.get("cnn_run") or info_va.get("mlp_run") != info.get("mlp_run"):
        raise RuntimeError("train and val feature files come from different runs. Set FORCE_REEXTRACT = True.")

    # Val metadata in the same order as the saved features (gives risk_level, site-unknown, etc. for the analysis)
    val_csv = pd.read_csv(VAL_METADATA_CSV)
    if val_csv["image"].duplicated().any():
        raise ValueError("duplicate image ids in the val metadata CSV")
    if not set(va["image"]) <= set(val_csv["image"]):
        raise ValueError("The saved val features contain images that are not in the val metadata CSV. "
                         "Set FORCE_REEXTRACT = True so they are rebuilt from the current CSV.")
    df = val_csv.set_index("image").loc[va["image"]].reset_index()
    if "risk_level" not in df.columns:
        df["risk_level"] = np.where(df["BinaryTarget"] == 1, "Malignant", "Benign")
    y = va["label"].astype(int)
    assert (df["BinaryTarget"].values.astype(int) == y).all(), "labels in the features don't match the CSV"
    groups = va["lesion_id_filled"]
    ytr = tr["label"].astype(int)
    print(f"\nVal images: {len(y)} ({y.mean():.1%} malignant) | train images: {len(ytr)} ({ytr.mean():.1%} malignant)")
    print(f"CNN: {info['cnn_model']} ({info['cnn_run']})\nMLP: {info['mlp_model']} ({info['mlp_run']})")

    # ---- scores on val ----
    scores = {"CNN only": softmax_malignant(va["cnn_logits"]), "MLP only": softmax_malignant(va["mlp_logits"])}
    fused_names, trained_heads = [], {}
    for head in HEADS:
        Xtr, Xva = inputs_for(head, tr), inputs_for(head, va)
        model = fit_head(head, Xtr, ytr)
        trained_heads[head] = model
        name = f"{HEAD_LABEL[head]} - fit on train"
        scores[name] = p_malignant(model, Xva)
        fused_names.append(name)
        name = f"{HEAD_LABEL[head]} - CV on val"
        scores[name] = cv_on_val(head, Xva, y, groups)
        fused_names.append(name)
        print(f"  built {HEAD_LABEL[head]} ({Xva.shape[1]} inputs)")

    auc_tr = roc_auc_score(ytr, margin(tr["cnn_logits"]))
    auc_va = roc_auc_score(y, margin(va["cnn_logits"]))
    print(f"\nCNN AUC on train images {auc_tr:.3f} vs val images {auc_va:.3f}. "
          + ("A big gap means the CNN's train outputs are overconfident, so trust the 'CV on val' rows more."
             if auc_tr - auc_va > 0.03 else "Small gap, so 'fit on train' and 'CV on val' should agree closely."))

    # ---- comparison table ----
    complete = va["complete_metadata"].astype(bool)
    site_unknown = (df["anatom_site_general_unknown"] == 1).values if "anatom_site_general_unknown" in df.columns \
        else np.zeros(len(df), bool)
    subsets = {"ALL val": np.ones(len(y), bool), "complete metadata": complete, "site unknown": site_unknown}
    boots = lesion_bootstrap_indices(groups, N_BOOT)
    ref = scores["CNN only"]

    rows = []
    for sub_name, mask in subsets.items():
        if mask.sum() < 50 or y[mask].min() == y[mask].max():
            continue
        for name, p in scores.items():
            row = {"subset": sub_name, "model": name, "n": int(mask.sum()), **headline(y[mask], p[mask])}
            if sub_name == "ALL val":
                row.update(bootstrap_auc(y, p, boots, ref=None if name == "CNN only" else ref))
            rows.append(row)
    comp = pd.DataFrame(rows)

    primary = PRIMARY
    if primary == "auto":
        primary = comp[(comp["subset"] == "ALL val") & comp["model"].isin(fused_names)].sort_values(
            "AUC", ascending=False)["model"].iloc[0]
    print(f"\nPrimary fused model (gets the image grid): {primary}"
          + ("  [picked by val AUC among fused models, so slightly optimistic]" if PRIMARY == "auto" else ""))

    # ---- output folder ----
    stamp = time.strftime("%Y-%m-%d_%H%M")
    out_root = RESULTS_ROOT / f"{stamp}_FUSION_{info['cnn_model']}+{info['mlp_model']}"
    out_root.mkdir(parents=True, exist_ok=True)

    comp.round(4).to_csv(out_root / "comparison.csv", index=False)
    show_cols = ["model", "n", "AUC", "AUC_lo", "AUC_hi", "dAUC_vs_CNN", "dAUC_lo", "dAUC_hi", "balanced_acc",
                 "malignant_recall", "benign_recall", "F2"] + [f"mal_recall@spec{s:.2f}" for s in SPEC_LEVELS]
    text = []
    for sub_name in subsets:
        sub = comp[comp["subset"] == sub_name]
        if sub.empty:
            continue
        cols = [c for c in show_cols if c in sub.columns and sub[c].notna().any()]
        text.append(f"=== {sub_name} (n={int(sub['n'].iloc[0])}; cutoff {CUTOFF:.2f} for recall/F2/accuracy) ===")
        text.append(sub[cols].round(3).to_string(index=False))
        text.append("")
    text.append("dAUC_vs_CNN = AUC(model) - AUC(CNN only) with a 95% CI from resampling whole lesions; "
                "if the CI includes 0 the difference is not clearly real.")
    (out_root / "comparison.txt").write_text("\n".join(text))
    print("\n" + "\n".join(text))

    # ---- matched-specificity per-diagnosis table ----
    from multiclass_analysis import RISK_GROUPED_ORDER
    mrows = []
    flagged = {n: p >= threshold_at_spec(y, p, MATCHED_SPEC) for n, p in scores.items()}
    for dx in RISK_GROUPED_ORDER:
        k = (df["diagnosis"] == dx).values
        if k.any():
            mrows.append({"diagnosis": dx, "n": int(k.sum()), **{n: float(f[k].mean()) for n, f in flagged.items()}})
    matched = pd.DataFrame(mrows)
    matched.to_csv(out_root / "matched_specificity_by_diagnosis.csv", index=False)
    print(f"Share of each diagnosis flagged malignant when every model is set to benign recall {MATCHED_SPEC:.2f} "
          "(malignant dx = recall; benign dx = false-alarm rate):")
    short = matched.rename(columns={n: n.replace(" - fit on train", " (tr)").replace(" - CV on val", " (cv)")
                                    for n in scores})
    print(short.round(3).to_string(index=False))

    # ---- per-model folders ----
    for name, p in scores.items():
        pred_path, summary, cm = analyze_model(name, p, df, y, out_root / slug(name))
        if name == primary:
            primary_pred, primary_cm = pred_path, cm
    print(f"\nConfusion matrix, primary model ({primary}), rows = true, columns = predicted [Benign, Malignant]:"
          f"\n{primary_cm}")

    if MAKE_IMAGE_GRID:
        try:
            from multiclass_analysis import plot_diagnosis_confidence_grid
            from config import LOCAL_IMAGE_DIR
            plot_diagnosis_confidence_grid(primary_pred, LOCAL_IMAGE_DIR, out_root / slug(primary) / "confidence_grid.png")
        except Exception as e:   # the grid is a nice-to-have; never lose the run over it
            print(f"(skipped image grid: {e})")

    # ---- save the heads trained on the train split ----
    import joblib
    (out_root / "fused_models").mkdir(exist_ok=True)
    for head, model in trained_heads.items():
        joblib.dump(model, out_root / "fused_models" / f"{head}_fit_on_train.joblib")

    (out_root / "run_summary.json").write_text(json.dumps({
        "cnn_run": info["cnn_run"], "mlp_run": info["mlp_run"], "cnn_model": info["cnn_model"],
        "mlp_model": info["mlp_model"], "heads": HEADS, "cutoff": CUTOFF, "n_boot": N_BOOT, "primary": primary,
        "n_val": int(len(y)), "n_train": int(len(ytr)), "cnn_auc_train": auc_tr, "cnn_auc_val": auc_va,
    }, indent=2))
    print(f"\nAll outputs saved to {out_root}/")


if __name__ == "__main__":
    main()