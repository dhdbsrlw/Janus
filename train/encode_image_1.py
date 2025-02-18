import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from janus.models import MultiModalityCausalLM, VLChatProcessor, VLMImageProcessor
from janus.utils.io import load_pil_images
import numpy as np
import os
import PIL.Image
from PIL import Image
import time

# img vocab size: 16384

# 1.
model_path = "/nas2/checkpoints/Janus-Pro-7B" # "deepseek-ai/Janus-1.3B"

vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
image_start_id = vl_chat_processor.image_start_id  # or however your code references it
image_end_id = vl_chat_processor.image_end_id      # or a custom token ID

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda() # .eval()


# 2.
IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)

image_processor = VLMImageProcessor(
        image_size=384, # 1024,
        image_mean=IMAGENET_INCEPTION_MEAN,
        image_std=IMAGENET_INCEPTION_STD,
        do_normalize=True,
    )

image = Image.open("/nas2/mllm_reasoning/sillm/official_seed/0204_flux/t2icompbench/gen/complex_val_rename/The blue mug is on top of the green coaster._68.png")
image_tensor = image_processor([image])
image_tensor = image_tensor['pixel_values'].to("cuda").to(torch.bfloat16) # torch.Size([1, 3, 384, 384])

output = vl_gpt.gen_vision_model.encode(
    image_tensor,  
)

# print("output: ", output) # (quant, emb_loss, info)
# quant: quantized continuous representation after passing through the VQ encoder + codebook. 
# [batch_size, codebook_embed_dim, height_q, width_q]
# torch.Size([1, 8, 24, 24])

# info: Any extra diagnostic info, especially the discrete code IDs used by the VQ encoder.


# 3. train sample
text_prompt = "User: The blue mug is on top of the green coaster.\nAssistant:<begin_of_image>"
text_ids = torch.LongTensor(tokenizer.encode(text_prompt)).to("cuda")
img_tokens = output[2][2] # .tolist()

# print("text_ids: ", text_ids)
# print("img_tokens: ", img_tokens)
# print(text_ids.shape)   # torch.Size([12])
# print(img_tokens.shape) # torch.Size([576])

full_ids = torch.cat((text_ids, img_tokens, torch.tensor([100593, 100001], device=text_ids.device)), dim=0)
# bos + prompt + boi + img + eoi + eos
print(full_ids)
print(full_ids.shape)

outputs = vl_gpt.language_model(
    input_ids=full_ids.unsqueeze(0),
    labels=full_ids.unsqueeze(0),
)
# outputs=vl_gpt.language_model(input_ids=full_ids.unsqueeze(0)) # bsz 1 추가
print(outputs.logits.shape) # torch.Size([1, 596, 102400]) # bsz, seq_len, _
print(outputs.loss) # None

