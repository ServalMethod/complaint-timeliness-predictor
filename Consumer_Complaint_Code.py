"""

Lucas Myler
10/04/2026
Foundations of Big Data Analytics CS356


================================================================================
CONSUMER COMPLAINT TIMELINESS PREDICTOR
================================================================================

PURPOSE:
    Predict whether a consumer complaint will receive a timely response from
    a company based on the CFPB Consumer Complaints dataset (complaints.csv).

APPROACH:
    This program implements a complete, minimal machine learning pipeline:

      1. INGEST
      2. CLEAN
      3. SPLIT
      4. TRAIN
      5. EVALUATE
      6. DIAGNOSE

KEY INSIGHT:
    This dataset is highly imbalanced (~99.4% "Yes"). Accuracy is therefore a
    misleading metric. The confusion matrix and F1 score expose how well the
    model actually identifies the rare "No" (late response) cases which is
    the classification task that matters most in practice.

================================================================================
"""




import csv
import random
from collections import Counter

# ============================================================
# 1. INGEST DATA
# ============================================================
def load_data(filename, max_rows=None):
    """Load CSV, keeping only the columns we need."""
    data = []
    keep = ['Product', 'Issue', 'Company', 'State', 
            'Submitted via', 'Timely response?']
    
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            data.append({k: row.get(k, '') for k in keep})
    return data

# ============================================================
# 2. CLEANSING AND PREPROCESSING
# ============================================================
def clean_data(data):
    """Remove rows with missing key values and normalize text."""
    cleaned = []
    for row in data:
        if not row['Product'] or not row['Timely response?']:
            continue
        for k in row:
            row[k] = row[k].strip()
        cleaned.append(row)
    return cleaned

# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================
def train_test_split(data, test_ratio=0.2, seed=42):
    """Shuffle data and split into train/test sets."""
    shuffled = data[:]
    random.seed(seed)
    random.shuffle(shuffled)
    split_index = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split_index], shuffled[split_index:]

# ============================================================
# 4. PREDICTIVE ALGORITHMS
# ============================================================
def train_by_product(data):
    groups = {}
    for row in data:
        groups.setdefault(row['Product'], []).append(row['Timely response?'])
    return {k: Counter(v).most_common(1)[0][0] for k, v in groups.items()}

def train_by_company(data):
    groups = {}
    for row in data:
        groups.setdefault(row['Company'], []).append(row['Timely response?'])
    return {k: Counter(v).most_common(1)[0][0] for k, v in groups.items()}

def train_by_product_issue(data):
    groups = {}
    for row in data:
        key = (row['Product'], row['Issue'])
        groups.setdefault(key, []).append(row['Timely response?'])
    return {k: Counter(v).most_common(1)[0][0] for k, v in groups.items()}

def predict(model, key, default="Yes"):
    return model.get(key, default)

# ============================================================
# 5. EVALUATION
# ============================================================
def evaluate(model, data, key_func):
    """Compute accuracy on the given data."""
    if not data:
        return 0.0
    correct = 0
    for row in data:
        if predict(model, key_func(row)) == row['Timely response?']:
            correct += 1
    return correct / len(data)

def baseline_accuracy(train_data, test_data):
    """Accuracy of always predicting the majority class from train."""
    if not train_data or not test_data:
        return 0.0
    majority = Counter(r['Timely response?'] for r in train_data).most_common(1)[0][0]
    correct = sum(1 for r in test_data if r['Timely response?'] == majority)
    return correct / len(test_data)

def confusion_matrix_and_metrics(model, data, key_func, positive="No"):
    """
    Compute confusion matrix and precision/recall/F1 for the minority class.
    
    Definitions (positive = "No" = late response):
      TP: actual=No,  predicted=No
      FP: actual=Yes, predicted=No
      FN: actual=No,  predicted=Yes
      TN: actual=Yes, predicted=Yes
    """
    TP = FP = FN = TN = 0
    for row in data:
        pred = predict(model, key_func(row))
        actual = row['Timely response?']
        
        if actual == positive and pred == positive:
            TP += 1
        elif actual != positive and pred == positive:
            FP += 1
        elif actual == positive and pred != positive:
            FN += 1
        else:
            TN += 1
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    
    return {
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def print_confusion_matrix(m):
    """Pretty-print a confusion matrix."""
    print(f"                     Predicted No     Predicted Yes")
    print(f"  Actual No          {m['TP']:>12,}     {m['FN']:>13,}")
    print(f"  Actual Yes         {m['FP']:>12,}     {m['TN']:>13,}")

def print_metrics(m, positive="No"):
    """Print precision, recall, F1 for the positive class."""
    print(f"\n  For class \"{positive}\" (late/non-timely response):")
    print(f"    Precision: {m['precision']:.2%}  "
          f"(of predicted '{positive}', how many were correct)")
    print(f"    Recall:    {m['recall']:.2%}  "
          f"(of actual '{positive}', how many we caught)")
    print(f"    F1 score:  {m['f1']:.2%}  "
          f"(harmonic mean of precision & recall)")

# ============================================================
# MAIN PROGRAM
# ============================================================
def main():
    filename = 'complaints.csv'
    MAX_ROWS = None
    TEST_RATIO = 0.2
    
    try:
        print(f"Loading data (max_rows={MAX_ROWS})...")
        raw_data = load_data(filename, max_rows=MAX_ROWS)
        print(f"  Loaded {len(raw_data):,} rows")
        
        print("Cleaning...")
        data = clean_data(raw_data)
        print(f"  After cleaning: {len(data):,} rows")
        
        if not data:
            print("No usable data found. Exiting.")
            return
        
        # ---- Split ----
        print(f"\nSplitting data ({int((1-TEST_RATIO)*100)}/{int(TEST_RATIO*100)} train/test)...")
        train_data, test_data = train_test_split(data, test_ratio=TEST_RATIO)
        print(f"  Train: {len(train_data):,} rows")
        print(f"  Test:  {len(test_data):,} rows")
        
        # ---- Class distribution ----
        train_dist = Counter(r['Timely response?'] for r in train_data)
        print(f"\nClass distribution (train):")
        for label, count in train_dist.most_common():
            print(f"  {label:5s}: {count:>12,} ({count/len(train_data):.2%})")
        
        # ---- Train models ----
        print("\nTraining models on training set...")
        model_product    = train_by_product(train_data)
        model_company    = train_by_company(train_data)
        model_prod_issue = train_by_product_issue(train_data)
        
        print(f"  Model A (Product):           {len(model_product):,} keys")
        print(f"  Model B (Company):           {len(model_company):,} keys")
        print(f"  Model C (Product + Issue):   {len(model_prod_issue):,} keys")
        
        # ---- Accuracy on test set ----
        print("\n" + "=" * 65)
        print("ACCURACY ON TEST SET")
        print("=" * 65)
        baseline = baseline_accuracy(train_data, test_data)
        print(f"  Baseline (always majority):       {baseline:.2%}")
        
        acc_a = evaluate(model_product, test_data, lambda r: r['Product'])
        print(f"  Model A (by Product):             {acc_a:.2%}")
        
        acc_b = evaluate(model_company, test_data, lambda r: r['Company'])
        print(f"  Model B (by Company):             {acc_b:.2%}")
        
        acc_c = evaluate(model_prod_issue, test_data,
                         lambda r: (r['Product'], r['Issue']))
        print(f"  Model C (by Product + Issue):     {acc_c:.2%}")
        
        # ---- Pick best model by accuracy ----
        candidates = [
            ("Product", model_product, lambda r: r['Product'], acc_a),
            ("Company", model_company, lambda r: r['Company'], acc_b),
            ("Product + Issue", model_prod_issue,
             lambda r: (r['Product'], r['Issue']), acc_c),
        ]
        best_name, best_model, best_key, best_acc = max(candidates, key=lambda t: t[3])
        
        # ---- Confusion matrix + metrics for best model ----
        print("\n" + "=" * 65)
        print(f"CONFUSION MATRIX — Best model: {best_name}")
        print("=" * 65)
        m = confusion_matrix_and_metrics(best_model, test_data, best_key)
        print_confusion_matrix(m)
        print_metrics(m, positive="No")
        
        # ---- Show sample predictions ----
        print("\n" + "=" * 65)
        print("SAMPLE PREDICTIONS FROM BEST MODEL")
        print("=" * 65)
        for row in test_data[:5]:
            key = best_key(row)
            pred = predict(best_model, key)
            actual = row['Timely response?']
            marker = "✓" if pred == actual else "✗"
            key_display = key if isinstance(key, str) else " / ".join(key)
            print(f"  {marker} {key_display[:55]:55s} -> pred={pred:3s} actual={actual}")
    
    except FileNotFoundError:
        print(f"Error: '{filename}' not found in this directory.")
        print("Make sure complaints.csv is in the same folder as this script.")
    except MemoryError:
        print("Out of memory! Try setting MAX_ROWS to a smaller number.")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
    
    
    
    
"""

RESULTS:
    
Loading data (max_rows=None)...
  Loaded 18,167,713 rows
Cleaning...
  After cleaning: 18,167,713 rows

Splitting data (80/20 train/test)...
  Train: 14,534,170 rows
  Test:  3,633,543 rows

Class distribution (train):
  Yes  :   14,445,896 (99.39%)
  No   :       88,274 (0.61%)

Training models on training set...
  Model A (Product):           21 keys
  Model B (Company):           7,810 keys
  Model C (Product + Issue):   342 keys

=================================================================
ACCURACY ON TEST SET
=================================================================
  Baseline (always majority):       99.39%
  Model A (by Product):             99.39%
  Model B (by Company):             99.51%
  Model C (by Product + Issue):     99.39%

=================================================================
CONFUSION MATRIX — Best model: Company
=================================================================
                     Predicted No     Predicted Yes
  Actual No                 5,598            16,475
  Actual Yes                1,419         3,610,051

  For class "No" (late/non-timely response):
    Precision: 79.78%  (of predicted 'No', how many were correct)
    Recall:    25.36%  (of actual 'No', how many we caught)
    F1 score:  38.49%  (harmonic mean of precision & recall)

=================================================================
SAMPLE PREDICTIONS FROM BEST MODEL
=================================================================
  ✓ EQUIFAX, INC.                                           -> pred=Yes actual=Yes
  ✓ EQUIFAX, INC.                                           -> pred=Yes actual=Yes
  ✓ BANK OF AMERICA, NATIONAL ASSOCIATION                   -> pred=Yes actual=Yes
  ✓ EQUIFAX, INC.                                           -> pred=Yes actual=Yes
  ✓ Experian Information Solutions Inc.                     -> pred=Yes actual=Yes
  
"""