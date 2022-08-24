import torch
import torchvision

import os
import torchvision.models as models
from torch.utils.mobile_optimizer import optimize_for_mobile

HERE = os.path.dirname(os.path.abspath(__file__))
model_path=os.path.join(HERE,'best_model_3_134258.pth')

checkpoint=torch.load(model_path)
model = models.resnet34(num_classes=4)
model.load_state_dict(checkpoint, strict=False)
model.eval()

# 근데 이 example은 왜 집어넣어봐야 되는거지?
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)

# 이래도 되는것 같기는 한데 무슨 차이가 있는지 찾아볼것
no_ex_module=torch.jit.script(model)
optimized_script_module=optimize_for_mobile(no_ex_module)
optimized_script_module._save_for_lite_interpreter('plz.ptl')
# traced_script_module_optimized = optimize_for_mobile(traced_script_module)
# traced_script_module_optimized._save_for_lite_interpreter("sb_lite_model.ptl")
