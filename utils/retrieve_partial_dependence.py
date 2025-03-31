from sklearn.inspection import partial_dependence


def retrieve_partial_dependences(model, X, meta_feature_names):
    
    partial_dependences = []
    
    for i, feature_name in enumerate(meta_feature_names):
        results = partial_dependence(model, X, [-i], grid_resolution=20)
        
        partial_dependences.append({
            "feature": feature_name,
            "partial_dependence": results
        })
    
    return results