"""
Author: gh Corgice @IceSandwich
Date: Febuary 2025
License: MIT
"""

#%%
from diffusers import StableDiffusionPipeline
#import torch
from safetensors import safe_open

# %%
class CheckpointPickle:
    import pickle

    load = pickle.load

    class Empty:
        pass

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            #TODO: safe unpickle
            if module.startswith("pytorch_lightning"):
                return CheckpointPickle.Empty
            return super().find_class(module, name)

file_path = "D:\GITHUB\stable-diffusion-webui\models\Stable-diffusion\dragonfruitUnisex_dragonfruitgtV10.safetensors"
with safe_open(file_path, framework="pt") as f:
    for x in f.keys():
        print(x)
    
# pipe = StableDiffusionPipeline.from_pretrained(R"D:\GITHUB\stable-diffusion-webui\models\Stable-diffusion\dragonfruitUnisex_dragonfruitgtV10.safetensors", use_safetensors=True)
# %%
