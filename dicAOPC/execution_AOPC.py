import global_variables as var
import AOPC
import AOPC_VGG16



for i in range(4):
    AOPC.AOPC_GradCAM(var.img_path, var.model_path, False)
    print("AOPC : {}".format(i))
    AOPC.AOPC_GradCAM_Random(var.img_path, var.model_path, False)
    print("AOPC_RANDOM : {}".format(i))
#AOPC_VGG16.computeDoubleAOPC("/mnt/data2/ImageNet/ILSVRC/Data/CLS-LOC/val")
