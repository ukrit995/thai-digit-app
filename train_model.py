import os
import json
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ─── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR  = 'dataset'
MODELS_DIR   = 'models'
RESULTS_DIR  = 'results'
MODEL_OUTPUT = 'models/current_model.pkl'
MODEL_INFO   = 'models/model_info.json'

CLASSES      = ['71', '72', '73', '74', '75']
THAI_LABELS  = ['๗๑', '๗๒', '๗๓', '๗๔', '๗๕']
IMG_SIZE     = (28, 28)
TEST_SIZE    = 0.2
RANDOM_STATE = 42


# ─── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(img_path):
    """โหลดและ preprocess ภาพ: grayscale → invert → crop → resize → normalize"""
    img = Image.open(img_path).convert('L')
    arr = np.array(img)

    if arr.mean() > 127:
        arr = 255 - arr

    arr = (arr > 30).astype(np.uint8) * 255

    rows = np.any(arr > 30, axis=1)
    cols = np.any(arr > 30, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        pad = 4
        rmin = max(0, rmin - pad)
        rmax = min(arr.shape[0], rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(arr.shape[1], cmax + pad)
        arr = arr[rmin:rmax+1, cmin:cmax+1]

    img_processed = Image.fromarray(arr).resize(IMG_SIZE, Image.LANCZOS)
    arr_final = np.array(img_processed, dtype=np.float32) / 255.0
    return arr_final


# ─── Load Dataset ──────────────────────────────────────────────────────────────
def load_dataset():
    X, y = [], []
    print("\n[LOAD] กำลังโหลด dataset...")
    for label_idx, cls in enumerate(CLASSES):
        folder = os.path.join(DATASET_DIR, cls)
        if not os.path.exists(folder):
            print(f"  [WARN] ไม่พบโฟลเดอร์: {folder}")
            continue
        images = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for fname in images:
            try:
                arr = preprocess_image(os.path.join(folder, fname))
                X.append(arr.flatten())
                y.append(label_idx)
            except Exception as e:
                print(f"  [SKIP] {fname}: {e}")
        print(f"  {THAI_LABELS[label_idx]} ({cls}): {len(images)} รูป")

    if len(X) == 0:
        raise ValueError("ไม่พบข้อมูลใน dataset/ กรุณาเพิ่มรูปภาพก่อน train")

    return np.array(X), np.array(y)


# ─── Define Models ─────────────────────────────────────────────────────────────
def get_models():
    return {
        'SVM': {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', SVC(kernel='rbf', C=10, gamma='scale',
                            probability=True, random_state=RANDOM_STATE))
            ]),
            'filename': 'svm_model.pkl',
            'description': 'SVM (RBF Kernel) + StandardScaler',
        },
        'Random Forest': {
            'model': Pipeline([
                ('clf', RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    min_samples_split=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                ))
            ]),
            'filename': 'random_forest_model.pkl',
            'description': 'Random Forest (200 trees)',
        },
        'KNN': {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', KNeighborsClassifier(
                    n_neighbors=5,
                    weights='distance',
                    metric='euclidean',
                    n_jobs=-1
                ))
            ]),
            'filename': 'knn_model.pkl',
            'description': 'KNN (k=5, distance-weighted) + StandardScaler',
        },
    }


# ─── Evaluate One Model ─────────────────────────────────────────────────────────
def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test, X_all, y_all):
    print(f"\n[TRAIN] {name} ...")
    pipeline.fit(X_train, y_train)
    print(f"  Train เสร็จสิ้น")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_all, y_all, cv=cv, scoring='accuracy')
    print(f"  CV Accuracy: {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")

    y_pred = pipeline.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"  Test Precision: {prec*100:.2f}%")
    print(f"  Test Recall   : {rec*100:.2f}%")
    print(f"  Test F1-Score : {f1*100:.2f}%")

    return {
        'name': name,
        'model': pipeline,
        'y_pred': y_pred,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'report': classification_report(y_test, y_pred, target_names=THAI_LABELS, zero_division=0),
        'cm': confusion_matrix(y_test, y_pred),
    }


# ─── Save Confusion Matrix ──────────────────────────────────────────────────────
def save_confusion_matrix(result, path):
    cm = result['cm']
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(THAI_LABELS)))
    ax.set_yticks(range(len(THAI_LABELS)))
    ax.set_xticklabels(THAI_LABELS, fontsize=14)
    ax.set_yticklabels(THAI_LABELS, fontsize=14)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(
        f"Confusion Matrix — {result['name']}  (Accuracy: {result['accuracy']*100:.1f}%)",
        fontsize=12
    )
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ─── Save Comparison Chart ──────────────────────────────────────────────────────
def save_comparison_chart(results, path):
    names    = [r['name'] for r in results]
    accs     = [r['accuracy'] * 100 for r in results]
    precs    = [r['precision'] * 100 for r in results]
    recs     = [r['recall'] * 100 for r in results]
    f1s      = [r['f1'] * 100 for r in results]
    cv_means = [r['cv_mean'] * 100 for r in results]
    cv_stds  = [r['cv_std'] * 100 for r in results]

    x = np.arange(len(names))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Comparison — Thai Digit Recognition (๗๑-๗๕)', fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.bar(x - 1.5*width, accs,  width, label='Accuracy',  color='#2563EB')
    ax.bar(x - 0.5*width, precs, width, label='Precision', color='#16A34A')
    ax.bar(x + 0.5*width, recs,  width, label='Recall',    color='#D97706')
    ax.bar(x + 1.5*width, f1s,   width, label='F1-Score',  color='#DC2626')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('Score (%)')
    ax.set_title('Test Set Metrics')
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in ax.patches:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f'{h:.1f}', ha='center', va='bottom', fontsize=8)

    ax2 = axes[1]
    colors = ['#2563EB', '#16A34A', '#D97706']
    bars = ax2.bar(x, cv_means, color=colors, width=0.5,
                   yerr=cv_stds, capsize=6, error_kw={'linewidth': 2})
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=11)
    ax2.set_ylabel('CV Accuracy (%)')
    ax2.set_title('5-Fold Cross-Validation Accuracy')
    ax2.set_ylim(0, 115)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    for bar, mean, std in zip(bars, cv_means, cv_stds):
        ax2.text(bar.get_x() + bar.get_width()/2, mean + std + 1,
                 f'{mean:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVE] บันทึกกราฟเปรียบเทียบที่ {path}")


# ─── Main ──────────────────────────────────────────────────────────────────────
def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X, y = load_dataset()
    print(f"\n[INFO] ข้อมูลทั้งหมด: {len(X)} ตัวอย่าง, {len(np.unique(y))} คลาส")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

    print("\n" + "="*60)
    print(" TRAINING 3 MODELS")
    print("="*60)

    model_configs = get_models()
    results = []

    for name, cfg in model_configs.items():
        result = evaluate_model(
            name, cfg['model'],
            X_train, X_test, y_train, y_test,
            X, y
        )
        result['description'] = cfg['description']
        result['filename']    = cfg['filename']
        results.append(result)

        model_path = os.path.join(MODELS_DIR, cfg['filename'])
        with open(model_path, 'wb') as f:
            pickle.dump(cfg['model'], f)
        print(f"  [SAVE] บันทึกไว้ที่ {model_path}")

    # ─── Compare ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Model':<18} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'CV Mean':>10}")
    print("-"*62)
    for r in results:
        print(f"{r['name']:<18} "
              f"{r['accuracy']*100:>9.2f}% "
              f"{r['precision']*100:>9.2f}% "
              f"{r['recall']*100:>9.2f}% "
              f"{r['f1']*100:>9.2f}% "
              f"{r['cv_mean']*100:>9.2f}%")

    best = max(results, key=lambda r: r['accuracy'])
    print(f"\n[BEST] โมเดลที่ดีที่สุด: {best['name']} (Accuracy = {best['accuracy']*100:.2f}%)")

    # ─── Save best as current_model.pkl ──────────────────────────────────────
    with open(MODEL_OUTPUT, 'wb') as f:
        pickle.dump(best['model'], f)
    print(f"[SAVE] บันทึก best model → {MODEL_OUTPUT}")

    # ─── Evaluation report ────────────────────────────────────────────────────
    lines = [
        "=== Thai Digit Recognition (๗๑-๗๕) — Evaluation Results ===",
        "",
        "Dataset",
        "-------",
        f"Total samples : {len(X)}",
        f"Train samples : {len(X_train)}",
        f"Test  samples : {len(X_test)}",
        f"Classes       : {', '.join(THAI_LABELS)}",
        "",
        "Model Comparison (Test Set)",
        "---------------------------",
        f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}",
        "-"*62,
    ]
    for r in results:
        lines.append(
            f"{r['name']:<20} "
            f"{r['accuracy']*100:>9.2f}% "
            f"{r['precision']*100:>9.2f}% "
            f"{r['recall']*100:>9.2f}% "
            f"{r['f1']*100:>9.2f}%"
        )
    lines += ["", f"Best Model: {best['name']} — {best['description']}", ""]

    for r in results:
        sep = "=" * 60
        lines += [
            sep,
            f"[{r['name']}] — {r['description']}",
            sep,
            f"Accuracy  : {r['accuracy']*100:.2f}%",
            f"Precision : {r['precision']*100:.2f}%",
            f"Recall    : {r['recall']*100:.2f}%",
            f"F1-Score  : {r['f1']*100:.2f}%",
            "",
            "Cross-Validation (5-Fold)",
            f"  Mean : {r['cv_mean']*100:.2f}%",
            f"  Std  : {r['cv_std']*100:.2f}%",
            f"  Folds: {[f'{s*100:.1f}%' for s in r['cv_scores']]}",
            "",
            "Classification Report:",
            r['report'],
            "Error Analysis:",
        ]
        errors = [(THAI_LABELS[y_test[i]], THAI_LABELS[r['y_pred'][i]])
                  for i in range(len(y_test)) if r['y_pred'][i] != y_test[i]]
        if errors:
            for (actual, pred), cnt in Counter(errors).most_common(5):
                lines.append(f"  จริง={actual} → ทำนาย={pred}: {cnt} ครั้ง")
        else:
            lines.append("  ไม่มี error ในชุดทดสอบ")
        lines.append("")

    with open(os.path.join(RESULTS_DIR, 'evaluation.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[SAVE] บันทึกผลการประเมินที่ {RESULTS_DIR}/evaluation.txt")

    # ─── Charts ───────────────────────────────────────────────────────────────
    save_confusion_matrix(best, os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    print(f"[SAVE] บันทึก confusion matrix ที่ {RESULTS_DIR}/confusion_matrix.png")

    save_comparison_chart(results, os.path.join(RESULTS_DIR, 'model_comparison.png'))

    # ─── model_info.json ──────────────────────────────────────────────────────
    with open(MODEL_INFO, 'w', encoding='utf-8') as f:
        json.dump({
            'name': 'current_model.pkl',
            'best_model': best['name'],
            'algorithm': best['description'],
            'accuracy': round(best['accuracy'] * 100, 2),
            'cv_accuracy': round(best['cv_mean'] * 100, 2),
            'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'all_models': [
                {
                    'name': r['name'],
                    'accuracy': round(r['accuracy'] * 100, 2),
                    'f1': round(r['f1'] * 100, 2),
                    'cv_accuracy': round(r['cv_mean'] * 100, 2),
                }
                for r in results
            ]
        }, f, ensure_ascii=False, indent=2)

    # ─── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f" DONE  Best model : {best['name']}")
    print(f"       Accuracy   : {best['accuracy']*100:.2f}%")
    print(f"       CV Acc     : {best['cv_mean']*100:.2f}% +/- {best['cv_std']*100:.2f}%")
    if best['accuracy'] >= 0.80:
        print("       [PASS] ผ่านเกณฑ์ขั้นต่ำ 80%")
    else:
        print("       [WARN] ยังไม่ถึง 80% — ลองเพิ่มข้อมูลหรือปรับ hyperparameter")
    print("="*60)

    return best['model']


if __name__ == '__main__':
    train()
