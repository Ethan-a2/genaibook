#!/usr/bin/env python
# coding: utf-8

# # Introduction

# This notebook is a supplementary material for the Introduction Chapter of the [Hands-On Generative AI with Transformers and Diffusion Models book](https://learning.oreilly.com/library/view/hands-on-generative-ai/9781098149239/).

# In[5]:


import diffusers
import huggingface_hub
import transformers

diffusers.logging.set_verbosity_error()
huggingface_hub.logging.set_verbosity_error()
transformers.logging.set_verbosity_error()


# ## Generating Images

# In[3]:


# from genaibook.core import get_device

# device = get_device()
# print(f"Using device: {device}")

import torch

# 1. 确定最佳可用设备
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    # MPS (Metal Performance Shaders) is for Apple Silicon GPUs
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")


# In[1]:


# import os
# proxy = 'http://127.0.0.1:7897'
# os.environ['HTTP_PROXY'] = proxy
# os.environ['HTTPS_PROXY'] = proxy


print(f"HTTP_PROXY set to: {os.environ.get('HTTP_PROXY', 'Not set')}")
print(f"HTTPS_PROXY set to: {os.environ.get('HTTPS_PROXY', 'Not set')}")
print(f"NO_PROXY set to: {os.environ.get('NO_PROXY', 'Not set')}")


# In[4]:


import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16",
).to(device)


# In[ ]:


prompt = "a photograph of an astronaut riding a horse"
pipe(prompt).images[0]


# In[6]:


import torch
torch.manual_seed(0)


# ## Generating Text

# In[8]:


from transformers import pipeline

classifier = pipeline("text-classification", device=device)
classifier("This movie is disgustingly good !")


# In[9]:


from transformers import set_seed

# Setting the seed ensures we get the same results every time we run this code
set_seed(10)


# In[10]:


generator = pipeline("text-generation")
prompt = "It was a dark and stormy"
generator(prompt)[0]["generated_text"]


# ## Generating Sound Clips

# In[11]:


pipe = pipeline("text-to-audio", model="facebook/musicgen-small", device=device)
data = pipe("electric rock solo, very intense")


# In[12]:


print(data)


# In[13]:


import IPython.display as ipd

display(ipd.Audio(data["audio"][0], rate=data["sampling_rate"]))


# In[ ]:




