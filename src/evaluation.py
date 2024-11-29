from sklearn.metrics import accuracy_score, f1_score

def classification_error(gt_targets, pred_targets):
    return 1 - accuracy_score(gt_targets[:, 0], pred_targets[:, 0])


def num_psc_decisions(gt_targets, pred_targets):
    # number of observations that were misclassified as poisonus, when in fact they are edible
    return sum((gt_targets[:, 1] == 0) & (pred_targets[:, 1] == 1))

def num_esc_decisions(gt_targets, pred_targets):
    # number of observations that were misclassified as edible, when in fact they are poisonus
    return sum((gt_targets[:, 1] == 1) & (pred_targets[:, 1] == 0))

    
def psc_esc_cost_score(gt_targets, pred_targets, cost_psc=100, cost_esc=1):
    return (cost_psc * num_psc_decisions(gt_targets, pred_targets) + cost_esc * num_esc_decisions(gt_targets, pred_targets))/ len(gt_targets)

def evaluate(gt_targets, pred_targets):
    # TODO modify this function so it can access the prediction and ground truth targets properly
    # this is just the first implementation
    if len(gt_targets) != len(pred_targets):
        raise ValueError('The number of ground truth targets and prediction targets must be equal.')

    class_err = classification_error(gt_targets, pred_targets)
    psc_esc_cost = psc_esc_cost_score(gt_targets, pred_targets)

    return {
        'classification_error': class_err,
        'f1_score': f1_score(gt_targets[:, 0], pred_targets[:, 0], average='macro'),
        'psc_esc_cost_score': psc_esc_cost,
        'user-focused_score': class_err + psc_esc_cost
    }

