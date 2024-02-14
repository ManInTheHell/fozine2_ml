This progect wroten for Gozine2 company entry AI project.

In this code we have Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) that are both metrics for evaluating the performance of machine learning models.

Use MAE:
When we want a metric that is more robust to outliers. MAE gives equal weight to all errors, regardless of their magnitude, making it less sensitive to outliers compared to RMSE.

Use RMSE:
When we want to penalize larger errors more heavily.
RMSE squares the errors before averaging them, which gives more weight to large errors compared to MAE. This can be useful when large errors are considered more detrimental to the model's performance.
