def select_best_model(results):
    return min(results, key=lambda x: results[x]["RMSE"])