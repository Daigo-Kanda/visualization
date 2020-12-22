import global_variables as var
import Grad_CAMforAOPC as gc

grad = gc.Grad_CAM("aaa", var.model_path)
grad.regression_get_npz(var.model_path, "/mnt/data2/img/20200209")
