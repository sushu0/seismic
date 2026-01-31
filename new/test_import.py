"""
40Hz模型评估和可视化 - 直接导入训练脚本的模型
"""
import sys
sys.path.insert(0, 'D:/SEISMIC_CODING/new')

# 先设置一些全局变量，避免训练脚本执行main
import builtins
builtins.__dict__['__name__'] = 'not_main'

# 直接导入训练脚本（但不执行main）
import importlib.util
spec = importlib.util.spec_from_file_location("train_40Hz", "D:/SEISMIC_CODING/new/train_40Hz_thinlayer.py")
train_module = importlib.util.module_from_spec(spec)

# 修改__name__避免执行main
train_module.__name__ = 'train_40Hz_imported'

try:
    spec.loader.exec_module(train_module)
except SystemExit:
    pass  # 如果脚本调用sys.exit，忽略

# 获取模型类
ThinLayerNetV2 = train_module.ThinLayerNetV2

print("模型类加载成功:", ThinLayerNetV2)
