import os
import sys
import torch

sys.path.append(os.getcwd())

print("Inspecting models/model_boxed.pth...")
try:
    checkpoint = torch.load("models/model_boxed.pth", map_location="cpu")
    print(f"Loaded object type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print("Checkpoint keys:")
        print(list(checkpoint.keys()))
        if "model_state_dict" in checkpoint:
            print("\nmodel_state_dict keys sample:")
            keys = list(checkpoint["model_state_dict"].keys())
            print(f"Total keys: {len(keys)}")
            for k in keys[:40]:
                print(f"  {k}: {checkpoint['model_state_dict'][k].shape}")
        else:
            print("\nkeys sample (direct state_dict):")
            print(list(checkpoint.keys())[:20])
    else:
        print("\nModel structure:")
        print(checkpoint)
except Exception as e:
    import traceback
    traceback.print_exc()
