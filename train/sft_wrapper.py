# pytorch lightning base SFT code
# conda activate janus2 (2호기)

import os
import torch
import numpy as np
import argparse
import pytorch_lightning as pl
from deepspeed.ops.adam import DeepSpeedCPUAdam
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
import transformers
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM
from PIL import Image

from janus.models import MultiModalityCausalLM, VLChatProcessor, VLMImageProcessor
from utils.config import build_config
from sft_datamodule import SFTDataModule
from lr_scheduler import CosineDecayWarmUpRestarts


class SFTWrapper(pl.LightningModule):
    def __init__(self, config, model, chat_processor, image_processor, tokenizer):
        super().__init__()
        self.config=config
        self.model=model
        self.chat_processor=chat_processor
        self.image_processor=image_processor
        self.tokenizer=tokenizer

        self.parallel_size = 1
        self.image_token_num_per_image = 576
        

    def setup(self, stage: str):
        self.model.train()


    def training_step(self, batch, batch_idx):
        combined_embeds, labels = self.prepare_input(batch)

        loss = self.forward(combined_embeds, labels)

        self.log_dict({'train/loss':loss,
                       'train/lr': self.trainer.optimizers[0].param_groups[0]['lr'],
                       'train/global_step': self.global_step}, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    

    def forward(self, combined_embeds, labels):
        outputs = self.model.language_model.model(inputs_embeds=combined_embeds)
        last_hidden_state = outputs.last_hidden_state
        # print(outputs.keys()) # odict_keys(['last_hidden_state', 'past_key_values'])
        # print("last_hidden_state: ", last_hidden_state.shape) # torch.Size([1, 595, 4096])

        logits = self.model.gen_head(last_hidden_state)
        # print("logits: ", logits.shape) # torch.Size([1, 595, 16384])

        # Flatten
        img_vocab_size = logits.size(-1)
        logits_2d = logits.view(-1, img_vocab_size)          # [seq_len, vocab_size]
        labels_1d = labels.view(-1)                          # [seq_len]

        loss = F.cross_entropy(logits_2d, labels_1d, ignore_index=-100)

        return loss


    def get_prompt(self, text):
        conversation = [
            {
                "role": "User",
                "content": text, # "The blue mug is on top of the green coaster.",
            },
            {"role": "Assistant", "content": ""},
        ]
        sft_format = self.chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + self.chat_processor.image_start_tag
        return prompt
    

    def get_text_embed(self, text):
        prompt = self.get_prompt(text)
        text_input_ids = self.tokenizer.encode(prompt)
        text_input_ids = torch.LongTensor(text_input_ids)

        text_tokens = torch.zeros((self.parallel_size, len(text_input_ids)), dtype=torch.int).cuda()
        for i in range(self.parallel_size):
            text_tokens[i, :] = text_input_ids # unsqueeze 효과

        text_input_embeds = self.model.language_model.get_input_embeddings()(text_tokens) # 텍스트 토큰을 임베딩으로 변환

        return text_input_embeds
    

    def get_image_embed(self, image: Image):
        image_tensor = self.image_processor([image])
        image_tensor = image_tensor['pixel_values'].to("cuda").to(torch.bfloat16) 
        # torch.Size([1, 3, 384, 384])

        output = self.model.gen_vision_model.encode(
            image_tensor,  
        )
        img_tokens = output[2][2]
        img_input_embeds = self.model.prepare_gen_img_embeds(img_tokens).unsqueeze(0)
        # img_embeds = img_embeds[:, :-1, :]
   
        # print(text_input_embeds.shape) # torch.Size([1, 19, 4096])
        # print(img_input_embeds.shape)    # torch.Size([1, 576, 4096])

        return img_tokens, img_input_embeds
    

    def prepare_input(self, batch):
        categories, captions, images, idxs = batch
        
        # for padding
        combined_embeds_list = []
        labels_list = []

        for text, image in zip(captions, images):
            text_embeds = self.get_text_embed(text)
            img_tokens, img_embeds = self.get_image_embed(image)

            # final input to the transformer for teacher-forcing:
            combined_embeds = torch.cat([text_embeds, img_embeds], dim=1)
            text_len = text_embeds.size(1)          # number of text tokens (19)
            image_len = img_embeds.size(1)            # 576
            labels = -100 * torch.ones(1, text_len + (image_len), dtype=torch.long).cuda()

            # We only fill labels for the image portion
            labels[:, text_len:] = img_tokens
            # print("labels: ", labels.shape) # torch.Size([1, 595])

            combined_embeds_list.append(combined_embeds)
            labels_list.append(labels)

        # find max length for padding
        max_seq_len = max(x.shape[1] for x in combined_embeds_list)
        batch_size = len(combined_embeds_list)
        embed_dim = combined_embeds_list[0].size(-1)

        # initialize
        padded_combined_embeds = torch.zeros(batch_size, max_seq_len, embed_dim).cuda()
        padded_labels = -100 * torch.ones(batch_size, max_seq_len, dtype=torch.long).cuda()

        # left padding
        for i, (embed, label) in enumerate(zip(combined_embeds_list, labels_list)):
            seq_len = embed.shape[1]
            pad_len = max_seq_len - seq_len 

            padded_combined_embeds[i, pad_len:, :] = embed  
            padded_labels[i, pad_len:] = label  # Right-align, pad left

        # return combined_embeds, labels
        return padded_combined_embeds, padded_labels


    @torch.no_grad()
    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)
    

    def on_before_optimizer_step(self, optimizer, _):
        # total grad norm
        self.log('train/grad_norm', self.compute_total_grad_norm(), 
                 on_step=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = DeepSpeedCPUAdam(
                    self.parameters(),
                    lr=self.config.optimizer.init_lr,
                    betas=self.config.optimizer.betas,
                    weight_decay=self.config.experiment.weight_decay
                )
        warmup_step = self.config.experiment.max_training_step * self.config.experiment.warmup_ratio
        scheduler = CosineDecayWarmUpRestarts(optimizer, 
                                              warmup_iter=warmup_step, 
                                              max_iter=self.config.experiment.max_training_step, 
                                              eta_min=self.config.optimizer.min_lr, 
                                              eta_max=self.config.optimizer.init_lr)
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
        }
        return [optimizer], [scheduler_config]
    
    def compute_total_grad_norm(self):
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm
    

def get_model(model_path, device):
    chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

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
    
    return vl_gpt, chat_processor, image_processor, tokenizer


def get_dataloader(config, tokenizer):
    # 학습 데이터: image/text pair 기준
    datamodule = SFTDataModule(config, tokenizer) 
    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = None # datamodule.val_dataloader() 

    return train_dataloader, val_dataloader


def get_trainer(config, device):
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=config.save_path, name=config.exp_name)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=tb_logger.log_dir,
        filename="{step:06d}",
        save_top_k=-1, # save all ckpt corresponding to saving interval              
        every_n_train_steps=config.experiment.save_steps, 
        # save_last=True
    )

    trainer = pl.Trainer(
        devices=config.world_size,
        accelerator=device,
        logger=tb_logger,
        default_root_dir=config.save_path,
        callbacks=[pl.callbacks.ModelSummary(max_depth=2), checkpoint_callback], 
        # strategy=DDPStrategy( # TODO: deepspeed                                          
        #     find_unused_parameters=False 
        # ),   
        strategy=DeepSpeedStrategy(
                stage=config.experiment.deepspeed.stage,                                                  
                allgather_bucket_size=config.experiment.deepspeed.allgather_bucket_size,
                reduce_bucket_size=config.experiment.deepspeed.reduce_bucket_size,
                offload_optimizer=config.experiment.deepspeed.offload_optimizer, 
                offload_parameters=config.experiment.deepspeed.offload_parameters,
                # pin_memory=config.experiment.deepspeed.pin_memory,
                # contiguous_gradients=config.experiment.deepspeed.contiguous_gradients,
                # overlap_comm=config.experiment.deepspeed.overlap_comm,
                # reduce_scatter=config.experiment.deepspeed.reduce_scatter,
                # allgather_partitions=config.experiment.deepspeed.allgather_partitions,
            ),         
        log_every_n_steps=config.experiment.log_steps,
        gradient_clip_val=config.experiment.gradient_clip_val, 
        enable_checkpointing=config.experiment.gradient_checkpointing,
        accumulate_grad_batches=config.experiment.gradient_accumulation_steps,
        precision="bf16" if config.precision is None or config.precision == "auto" else config.precision, #config.precision, 
        max_steps=config.experiment.max_training_step, # or max_epochs   
        check_val_every_n_epoch=None,
        val_check_interval=config.experiment.val_steps * config.experiment.gradient_accumulation_steps, 
        # num_sanity_val_steps = 0,           
    )

    return trainer


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="configs/train.yaml")
    args = parser.parse_args()
    config = build_config(cfg_path=args.cfg_path)
    return config
    

def main(config):
    if config.save_path is not None:
        os.makedirs(config.save_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pl.seed_everything(config.experiment.seed, workers=True) 
    
    model, chat_processor, image_processor, tokenizer = get_model(config.model_path)
    
    train_dataloader, val_dataloader = get_dataloader(config, tokenizer) 

    wrapper = SFTWrapper(config, 
                         model=model, 
                         chat_processor=chat_processor, 
                         image_processor=image_processor,
                         tokenizer=tokenizer) 
    wrapper.model.print_trainable_parameters()
    trainer = get_trainer(config, device)
    
    trainer.fit(wrapper, train_dataloaders=train_dataloader) # TODO: 일단 val_dataloader 는 사용하지 않음.



if __name__ == "__main__":

    main(config=load_config())