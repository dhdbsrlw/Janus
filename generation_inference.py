# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import os
import PIL.Image
import time

# specify the path to the model
model_path = "/nas2/checkpoints/Janus-Pro-7B" # "deepseek-ai/Janus-1.3B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

conversation = [
    {
        "role": "User",
        "content": "A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue.",
    },
    {"role": "Assistant", "content": ""},
]

sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation,
    sft_format=vl_chat_processor.sft_format,
    system_prompt="",
)
prompt = sft_format + vl_chat_processor.image_start_tag

# print("\ndebug 1")
# print(vl_chat_processor.sft_format)
# print("sft_format: ", sft_format) 
# print("prompt: ", prompt)

"""
debug 1
deepseek
sft_format:  User: A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue.

Assistant:
prompt:  User: A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue.

Assistant:<begin_of_image>
"""

@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 1, # 16,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    # print("\ndebug 2")
    # print("input_ids before tensor: ", input_ids)
    input_ids = torch.LongTensor(input_ids)
    # print("input_ids after being tensor: ", input_ids)
    # print(input_ids.shape)

    """
    debug 2
    input_ids before tensor:  [100000, 5726, 25, 338, 3415, 12, 394, 1461, 12, 64548, 8072, 280, 20560, 28799, 5989, 9368, 2112, 276, 427, 96575, 18972, 11, 1090, 245, 5501, 2653, 9539, 280, 813, 5969, 3164, 11, 87607, 10421, 7524, 11, 285, 14790, 1130, 1971, 348, 2735, 280, 5501, 13, 185, 185, 77398, 25, 100016]
    input_ids after being tensor:  tensor([100000,   5726,     25,    338,   3415,     12,    394,   1461,     12,
            64548,   8072,    280,  20560,  28799,   5989,   9368,   2112,    276,
            427,  96575,  18972,     11,   1090,    245,   5501,   2653,   9539,
            280,    813,   5969,   3164,     11,  87607,  10421,   7524,     11,
            285,  14790,   1130,   1971,    348,   2735,    280,   5501,     13,
            185,    185,  77398,     25, 100016]) # 100016: boi
    torch.Size([50])
    """

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    # print("\ndebug 3")
    # print("tokens: ", tokens)
    # print(tokens.shape) # torch.Size([2, 50])

    for i in range(parallel_size*2):
        # 짝수 인덱스: actual prompt 내용을 담은 conditioned input
        # 홀수 인덱스: unconditioned input (w/ masked-out tokens)
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id # padding token 으로 모두 mask out
    
    # print("\ndebug4")
    # print("after making tokens: ", tokens)
    # print(tokens.shape)

    """
    after making tokens:  tensor([[100000,   5726,     25,    338,   3415,     12,    394,   1461,     12,
          64548,   8072,    280,  20560,  28799,   5989,   9368,   2112,    276,
            427,  96575,  18972,     11,   1090,    245,   5501,   2653,   9539,
            280,    813,   5969,   3164,     11,  87607,  10421,   7524,     11,
            285,  14790,   1130,   1971,    348,   2735,    280,   5501,     13,
            185,    185,  77398,     25, 100016],
        [100000, 100015, 100015, 100015, 100015, 100015, 100015, 100015, 100015,
         100015, 100015, 100015, 100015, 100015, 100015, 100015, 100015, 100015,
         100015, 100015, 100015, 100015, 100015, 100015, 100015, 100015, 100015,
         100015, 100015, 100015, 100015, 100015, 100015, 100015, 100015, 100015,
         100015, 100015, 100015, 100015, 100015, 100015, 100015, 100015, 100015,
         100015, 100015, 100015, 100015, 100016]], device='cuda:0',
       dtype=torch.int32)
    torch.Size([2, 50])
    """

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens) # 텍스트 토큰을 임베딩으로 변환
    # print("\ndebug 5")
    # print("inputs_embeds: ", inputs_embeds)
    # print(inputs_embeds.shape)

    """
   debug 5
    inputs_embeds:  tensor([[[-1.0757e-03,  1.0538e-04,  2.0117e-01,  ..., -1.1749e-03,
          -3.0899e-04, -1.1368e-03],
         [-4.3640e-03, -7.9956e-03,  1.9824e-01,  ...,  1.0889e-01,
           1.0742e-01, -1.5332e-01],
         [-5.9204e-03,  6.3477e-02,  3.3789e-01,  ...,  4.7607e-02,
           3.0029e-02, -1.3657e-03],
         ...,
         [ 1.3184e-01,  1.2988e-01, -5.8838e-02,  ...,  8.3496e-02,
          -2.2656e-01,  3.4668e-02],
         [-5.9204e-03,  6.3477e-02,  3.3789e-01,  ...,  4.7607e-02,
           3.0029e-02, -1.3657e-03],
         [ 2.9907e-02, -1.0254e-02,  4.2969e-02,  ..., -4.3701e-02,
           3.8574e-02, -8.0566e-03]],

        [[-1.0757e-03,  1.0538e-04,  2.0117e-01,  ..., -1.1749e-03,
          -3.0899e-04, -1.1368e-03],
         [ 1.3428e-03, -5.2795e-03,  2.8809e-02,  ...,  1.5076e-02,
          -3.1433e-03, -8.9111e-03],
         [ 1.3428e-03, -5.2795e-03,  2.8809e-02,  ...,  1.5076e-02,
          -3.1433e-03, -8.9111e-03],
         ...,
         [ 1.3428e-03, -5.2795e-03,  2.8809e-02,  ...,  1.5076e-02,
          -3.1433e-03, -8.9111e-03],
         [ 1.3428e-03, -5.2795e-03,  2.8809e-02,  ...,  1.5076e-02,
          -3.1433e-03, -8.9111e-03],
         [ 2.9907e-02, -1.0254e-02,  4.2969e-02,  ..., -4.3701e-02,
           3.8574e-02, -8.0566e-03]]], device='cuda:0', dtype=torch.bfloat16)
    torch.Size([2, 50, 4096])
    """

    # (앞으로 생성될) 이미지 토큰의 저장 공간 미리 할당
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda() 
    # print("generated_tokens: ", generated_tokens)
    # print(generated_tokens.shape) # torch.Size([1, 576])

    for i in range(image_token_num_per_image):
        # 생성
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state # torch.Size([2, 1, 4096])
        
        # CFG 적용
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond) # 최종 로짓
        probs = torch.softmax(logits / temperature, dim=-1)

        # 위 prob 으로부터 다음 이미지 토큰 생성
        next_token = torch.multinomial(probs, num_samples=1) # probability distribution 에서 토큰을 샘플링
        generated_tokens[:, i] = next_token.squeeze(dim=-1)
        # print("next_token: ", next_token) # tensor([[8920]], device='cuda:0')
        # print(next_token.shape) # torch.Size([1, 1])
        # print(generated_tokens)
        # """
        # tensor([[  247,  2059, 11399, 14965,  6145, 14191, 16336,  8804,  3315, 15586,
        #    939,  4572, 13989,  1358,  3808,  4350, 13603, 15527, 11627, 11167,
        #   6013,  6078,  1285, 16104,  6111,  8408, 14380, 10132,  4936, 14855,
        #   4828,  2900,  9245,  1695, 10924, 12873, 16239,  7750,  3475,   584,
        #   9128, 15672, 14200, 14454,  2608, 11182, 10924,  8314,  6236, 15400,
        #   7438,  1646, 14902,  8183,  4943, 11701,  1061, 10675,   566, 10825,
        #   8974, 14912,  2871,  9595,  1745, 13954, 15915,  3081, 12777,  5946,
        #  16066,  8299,  1727, 15557, 13934,  5630, 10365,  7970,  4341,   953,
        #  13580,  2428, 14733,  6651, 12527, 16223,  3583,  4059,  9805, 14855,
        #   5497,  2446,  3018,  7899, 12843, 14795, 12657,  6836,   875,   882,
        #  11260,  1113,  3615, 12430,  3416,  6893,  5736,  4743, 13146,  8671,
        #   4159,  5365,  4869,  5427,  7721, 13128, 11493,  7371,  2131,  4914,
        #    408, 16179, 14877,    60,  7997, 12323,  1713,  3381,  6070,  5543,
        #  15769, 13233, 14661,  9475,  3843, 16320, 14386, 12772,   888, 12947,
        #  16177, 13222,  1201, 15680,  8151,  4463,  4704,  2268,  1031, 15815,
        #  12362,  7156, 13850, 13123,  1461,  5983,  8668, 16328, 10838, 13803,
        #  10733,  7386,  1302,  3732, 11853,  6240,  8516,  4164,  3281,  3813,
        #   4390,  1614,  3058,  7360,   386, 12870, 13745, 16121,  5594,  9144,
        #  13637, 10700, 11354,  1995,  9970,    24, 15019,  9204,  6878,  9025,
        #   5764,  7254,  9900,  9593, 12191,    37,  7819, 14296,  9432, 16264,
        #    534, 10564, 10297,  9169,  4042, 10043,   773, 15650, 13074,  9439,
        #   2131,  4512, 14434, 15972,  2479,  3809, 15064,  7639, 15118,  2190,
        #    920, 12455,  4006, 13603,  1510, 14062, 13860, 11202,  7870, 11233,
        #  14373,  7245,  9246,  4271, 10804,   743,   589,  5841,  5855,  5915,
        #   3951,  5227,  5431,  3840,  5905,  5984,  8216, 15598,  8269,  9471,
        # """
        # print(generated_tokens.shape) # torch.Size([1, 576])

        # 그 다음 생성을 위한 준비
        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1) # torch.Size([2])
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token) # torch.Size([2, 4096])
        inputs_embeds = img_embeds.unsqueeze(dim=1) # torch.Size([2, 1, 4096])

    # 8-channel latent representation
    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    """
    dec 1:  tensor([[[[-0.9531, -0.9648, -0.9844,  ..., -1.0000, -1.0391, -0.9805],
          [-0.9453, -1.0156, -0.9961,  ..., -1.0312, -1.0156, -1.0078],
          [-1.0156, -1.0000, -0.9961,  ..., -1.0234, -1.0000, -1.0078],
          ...,
          [-1.0000, -1.0156, -1.0078,  ...,  0.6914,  0.8359,  0.7656],
          [-0.9961, -1.0156, -0.9922,  ...,  0.3398,  0.6328,  0.7344],
          [-1.0000, -1.0391, -1.0312,  ..., -0.0786, -0.0723, -0.0050]],
    torch.Size([1, 3, 384, 384])   
    """
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255) # 0 - 255 사이의 픽셀값으로 변환
    """
    dec 3:  [[[[  5.9765625    9.9609375   37.353516  ]
   [  4.482422     9.462891    39.84375   ]
   [  1.9921875   14.443359    38.34961   ]
   ...
   [  0.          24.902344    72.46582   ]
   [  0.          20.419922    70.22461   ]
   [  2.4902344   23.90625     62.753906  ]]
   (1, 384, 384, 3)
    """
 
    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs('generated_samples', exist_ok=True)
    for i in range(parallel_size):
        save_path = os.path.join('generated_samples', "img_{}.jpg".format(i))
        PIL.Image.fromarray(visual_img[i]).save(save_path)

start = time.time() 

generate(
    vl_gpt,
    vl_chat_processor,
    prompt,
)

end = time.time()
# print("time elapsed: ", end - start)
# 이미지 1개 기준 13초 소요 (모델/토크나이저 로드 제외)