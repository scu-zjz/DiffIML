import torch

weight_path = 'MantraNetv4.pt'
state_dict = torch.load(weight_path)

new_state_dict =  {'model': {}}
for key, value in state_dict.items():
    new_state_dict['model'][key] = value

new_weight_path = '/mnt/data0/dubo/workspace/IMDLBenCo/ckpt/IMDLBenCo_ckpt/MantraNet_pretrain.pth'
torch.save(new_state_dict, new_weight_path)

print(f"Modified weights saved to {new_weight_path}")
