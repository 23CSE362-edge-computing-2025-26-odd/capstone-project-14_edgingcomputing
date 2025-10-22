import numpy as np

# This is a Fuzzy Inference System (FIS) using numpy.

# 1. Membership Functions (Triangular) - Robust version
def trimf(x, params):
    a, b, c = params
    epsilon = 1e-9
    term1 = (x - a) / (b - a + epsilon)
    term2 = (c - x) / (c - b + epsilon)
    return np.maximum(0, np.minimum(term1, term2))

# 2. Fuzzification: Get the degree of membership for an input value
def fuzzify_zscore(z_score):
    return {
        "Low": trimf(z_score, [-1, 0, 2.0]),
        "Medium": trimf(z_score, [1.5, 3.0, 4.5]),
        "High": trimf(z_score, [3.5, 5.0, 100.0])
    }

# 3. Rule Base & Inference
def apply_rules(fuzzified_inputs):
    # Rules: IF ZScore is X, THEN Anomaly is Y
    return {
        "Low": fuzzified_inputs["Low"],
        "Medium": fuzzified_inputs["Medium"],
        "Critical": fuzzified_inputs["High"]
    }

# 4. Defuzzification (Centroid Method)
def defuzzify(implicated_outputs):
    output_domain = np.arange(0, 1.01, 0.01)
    output_mf = {
        "Low": trimf(output_domain, [-0.1, 0, 0.4]),
        "Medium": trimf(output_domain, [0.3, 0.5, 0.7]),
        "Critical": trimf(output_domain, [0.6, 1.0, 1.1])
    }
    
    clipped_mfs = [np.minimum(implicated_outputs[level], output_mf[level]) for level in implicated_outputs]
    aggregate_mf = np.maximum.reduce(clipped_mfs)
    
    numerator = np.sum(output_domain * aggregate_mf)
    denominator = np.sum(aggregate_mf)
    
    if denominator == 0:
        return 0.0
        
    return numerator / denominator

# Main function to run the fuzzy logic
def get_anomaly_score(metrics):
    if not isinstance(metrics, list) or not metrics:
        return 0.0

    arr = np.array(metrics)
    median = np.median(arr)
    mad = np.median(np.abs(arr - median)) + 1e-9
    z_scores = 0.6745 * np.abs(arr - median) / mad
    max_z_score = np.max(z_scores)

    fuzzified_z = fuzzify_zscore(max_z_score)
    rule_outputs = apply_rules(fuzzified_z)
    anomaly_score = defuzzify(rule_outputs)
    
    return anomaly_score