from regresion import *

result = {}
mae, root_mse = linear()
result['linear'] = {"MAE": mae, "RMSE": root_mse}
print(f'linear model: MAE = {mae}, RMSE = {root_mse}')

mae, root_mse = ridge()
result['ridge'] = {"MAE": mae, "RMSE": root_mse}
print(f'ridge model: MAE = {mae}, RMSE = {root_mse}')

mae, root_mse = lasso()
result['lasso'] = {"MAE": mae, "RMSE": root_mse}
print(f'lasso model: MAE = {mae}, RMSE = {root_mse}')

mae, root_mse = elastic_net()
result['elastic_net'] = {"MAE": mae, "RMSE": root_mse}
print(f'elastic_net model: MAE = {mae}, RMSE = {root_mse}')

mae, root_mse = svr()
result['svr'] = {"MAE": mae, "RMSE": root_mse}
print(f'svr model: MAE = {mae}, RMSE = {root_mse}')

min_rmse_algorithm = min(result, key=lambda x: result[x]["RMSE"])
min_mae_algorithm = min(result, key=lambda x: result[x]["MAE"])
print('------------------------')
print("Algorithm with minimum MAE:", min_mae_algorithm)
print("Algorithm with minimum RMSE:", min_rmse_algorithm)
