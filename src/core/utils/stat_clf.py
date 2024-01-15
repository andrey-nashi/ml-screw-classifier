

def ml_stat_binary_classification(tp: int, tn: int, fp: int, fn: int):
    output = {}

    # ---- TPR, Sensitivity
    if tp + fn != 0: output["sensitivity"] = tp / (tp + fn)
    else: output["sensitivity"] = -1

    # ---- SPC, Specificity
    if fp + tn != 0: output["specificity"] = tn / (fp + tn)
    else: output["specificity"] = -1

    # ---- PPV, Precision
    if tp + fp != 0: output["precision"] = tp / (tp + fp)
    else: output["precision"] = - 1

    # ---- Accuracy
    if tp + fp + tn + fn != 0: output["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
    else: output["accuracy"] = -1

    # ---- F1 Scrore
    if 2 * tp + fp + fn != 0: output["f1_score"] = 2 * tp / (2 * tp + fp + fn)
    else: output["f1_score"] = -1

    return output

