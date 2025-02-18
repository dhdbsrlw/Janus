import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from janus.models import MultiModalityCausalLM, VLChatProcessor, VLMImageProcessor

from janus.utils.io import load_pil_images
import numpy as np
import os
from PIL import Image
import time

# img vocab size: 16384

# 1.
model_path = "/nas2/checkpoints/Janus-Pro-7B" # "deepseek-ai/Janus-1.3B"

vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().train() # no eval()


# 2. 
conversation = [
    {
        "role": "User",
        "content": "The blue mug is on top of the green coaster.",
    },
    {"role": "Assistant", "content": ""},
]

sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation,
    sft_format=vl_chat_processor.sft_format,
    system_prompt="",
)
prompt = sft_format + vl_chat_processor.image_start_tag


# 3. 
parallel_size = 1
image_token_num_per_image = 576

input_ids = tokenizer.encode(prompt)
input_ids = torch.LongTensor(input_ids)

tokens = torch.zeros((parallel_size, len(input_ids)), dtype=torch.int).cuda() # parallel_size = 1 로 대체
for i in range(parallel_size):
    tokens[i, :] = input_ids # unsqueeze 효과

inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens) # 텍스트 토큰을 임베딩으로 변환

generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda() 



# 4.
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
img_tokens = output[2][2]
img_embeds = vl_gpt.prepare_gen_img_embeds(img_tokens).unsqueeze(0)
# img_embeds = img_embeds[:, :-1, :]
# print(img_tokens[-1]) # tensor(7969, device='cuda:0')


# print("inputs_embeds:", inputs_embeds)
# print("img_embeds:", img_embeds)
print(inputs_embeds.shape) # torch.Size([1, 19, 4096])
print(img_embeds.shape)    # torch.Size([1, 576, 4096])


# 5.
# final input to the transformer for teacher-forcing:
combined_embeds = torch.cat([inputs_embeds, img_embeds], dim=1)
text_len = inputs_embeds.size(1)          # number of text tokens (19)
image_len = img_embeds.size(1)            # 576
labels = -100 * torch.ones(1, text_len + (image_len), dtype=torch.long).cuda()

# We only fill labels for the image portion
# The image portion in combined_embeds starts at index text_len, and has length (image_len - 1).
labels[:, text_len:] = img_tokens # img_tokens[1:] # img_tokens[:, 1:] 
print("labels: ", labels)
print("labels: ", labels.shape) # torch.Size([1, 595])


# 6.
# try 1
# outputs = vl_gpt.language_model.model(
#     inputs_embeds=combined_embeds,
#     labels=labels
# )
# # Traceback (most recent call last):
# # TypeError: forward() got an unexpected keyword argument 'labels'
# loss = outputs.loss
# print("loss:", loss)


# try 2
outputs = vl_gpt.language_model.model(inputs_embeds=combined_embeds)
last_hidden_state = outputs.last_hidden_state
# print(outputs.keys()) # odict_keys(['last_hidden_state', 'past_key_values'])
print("last_hidden_state: ", last_hidden_state.shape) # torch.Size([1, 595, 4096])

logits = vl_gpt.gen_head(last_hidden_state)
# print(logits)
print("logits: ", logits.shape) # torch.Size([1, 595, 16384])

img_vocab_size = logits.size(-1)
# Flatten
logits_2d = logits.view(-1, img_vocab_size)          # [seq_len, vocab_size]
labels_1d = labels.view(-1)                          # [seq_len]

loss = F.cross_entropy(logits_2d, labels_1d, ignore_index=-100)
print("loss:", loss)
