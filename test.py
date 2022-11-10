import options
import utils

cfg = options.get_arguments()
model = utils.build_structure_generator(cfg)
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print("Total trainable params:", total_trainable_params)
print("Total params:", total_params)