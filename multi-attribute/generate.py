# Adapted from https://github.com/huggingface/transformers/blob/21da895013a95e60df645b7d6b95f4a38f604759/examples/run_glue.py and https://github.com/huggingface/transformers/blob/21da895013a95e60df645b7d6b95f4a38f604759/examples/run_generation.py


import argparse
import glob
import json
import logging
import os

import time
import numpy as np
import torch

from tqdm import tqdm, trange
import sys
import codecs
import math

from modeling_gpt2 import GPT2LMHeadModel

from transformers import (
    GPT2Config,
    GPT2Tokenizer
)


import os.path
import time

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
}
#GPT2 added as per - https://huggingface.co/transformers/model_doc/gpt2.html
def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--model_type",default="gpt2",type=str,help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),)
    parser.add_argument( "--gen_model_name_or_path",default="gpt2-xl",type=str,help="Path to language model (usually `gpt2-xl' can also use `gpt2-large', `gpt2-medium', `gpt2')",)
    parser.add_argument("--gedi_model_name_or_path",default=None,type=str,help="Path to GeDi model. Assumes path from --mode if set to None"),
    parser.add_argument("--fp16", action="store_true", default=False, help="convert GPT-2 XL weights to fp16 (saves signifcant GPU RAM). May sometimes change samples slighty.")
    parser.add_argument("--load_in_half_prec",action="store_true", help="loads in half precision to save CPU memory. May sometimes change samples slighty.")

    # Other parameters
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",)
    parser.add_argument("--tokenizer_name",default="",type=str,help="Pretrained tokenizer name or path if not the same as model_name",)
    parser.add_argument("--cache_dir",default="",type=str,help="Where do you want to store the pre-trained models downloaded from s3",)
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    ##arguments for generation
    parser.add_argument("--gen_length", type=int, default=128, help= "generation length")
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")
    parser.add_argument("--temperature", type=float, default=1.0,help="lower tend toward greedy sampling",)
    parser.add_argument("--disc_weight", type=float, default=30.0,help="weight for GeDi discriminator",)
    parser.add_argument("--sim_weight", type=float, default=10.0,help="weight for GeDi discriminator",)
    parser.add_argument("--filter_p", type=float, default=0.8,help="filters at up to filter_p cumulative probability from next token distribution.")
    parser.add_argument("--class_bias", type=float, default=None,help="biases GeDi's classification probabilties")
    parser.add_argument("--target_p", type=float, default=0.8,help="In comination with filter_p, saves tokens with above target p probability of being in the correct class ")
    parser.add_argument("--do_sample", action="store_true",help="If set to False greedy decoding is used. Otherwise sampling is used. Defaults to True.")
    parser.add_argument("--do_sim", action="store_true",help="If set to False no calulate similarity. Otherwise calulate similarity is used. Defaults to True.")
    parser.add_argument("--repetition_penalty",default=1.2,type=float,help="The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.")
    parser.add_argument("--rep_penalty_scale",default=10.0,type=float,help="can use positive number to set max logit. rep penalty works better with positive logits")
    parser.add_argument("--penalize_cond", action="store_true",help="apply repetition penalty to tokens in coditioning text")
    parser.add_argument("--k", type=float, default=None,help="top_k,The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.",)
    parser.add_argument("--p", type=float, default=0.8,help="top_p,The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.",)

    parser.add_argument("--gen_type", type=str, default="gedi", help="gedi, cclm, or gpt2")
    parser.add_argument("--secondary_code", type=str, default="film", help="secondary topic control code")
    parser.add_argument("--gpt3_api_key", type=str, default=None,  help= "GPT-3 api key" )
    parser.add_argument("--sentiment_code", type=str, default="positive", help="positive or negative")

    parser.add_argument("--prompt", type=str, default="",  help= "prompt for generation" )

    args = parser.parse_args()
    assert(args.gen_type=="gedi" or args.gen_type=="cclm" or args.gen_type=="gpt2")

    args.code_desired = "true"
    args.code_undesired = "false"
    if not args.gen_type == "gpt2":
        if args.gedi_model_name_or_path is None:
            args.gedi_model_name_or_path = "./pretrained_models/gedi_topic"
            if not os.path.isdir(args.gedi_model_name_or_path):
                raise Exception("GeDi model path not found, must either run `get_models.sh' or set `args.gedi_model_name_or_path'")
    if args.class_bias is None:
        args.class_bias = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES["gpt2"]

    tokenizer = tokenizer_class.from_pretrained(args.gen_model_name_or_path,do_lower_case=False)


    if not(args.gpt3_api_key is None):
        print("It's a bit wasteful but our code needs to load GPT-2 even if using GPT-3")
    if args.load_in_half_prec:  # false
        model = model_class.from_pretrained(args.gen_model_name_or_path,load_in_half_prec=True)
        model.to(args.device)
        #even if using --fp16, apex doesn't accept half prec models, so requires converting to full prec before converting back
        model = model.float()
    else:
        model = model_class.from_pretrained(args.gen_model_name_or_path)
        model.to(args.device)

    #using fp16 option with GPT2-XL can prevent OOM errors. OOM(out of memory)
    if args.fp16:   # False
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        #opt_level O3 gives fully half precision weights. Okay for GPT-2 as long as we don't need to finetune
        model = amp.initialize(model, opt_level='O3')
        torch.cuda.empty_cache()

    args.length = adjust_length_to_model(args.gen_length,max_sequence_length=1024)

    if args.gen_type == "gedi" :
        gedi_model = model_class.from_pretrained(args.gedi_model_name_or_path)
        gedi_model.to(args.device)
    else:
        gedi_model=None

    user_prompt = len(args.prompt)==0
    while True:
        if user_prompt:
            while True:
                code = input("Give a secondary code or hit enter to keep as " + str(args.secondary_code) + ": ")
                sentiment = input("Give a sentiment or hit enter to keep ad " + str(args.sentiment_code) + ": ")
                if (len(code)==0) and (len(sentiment)==0):
                    break
                elif len(sentiment)==0:
                    args.secondary_code = code
                    break
                elif len(code)==0:
                    args.sentiment_code = sentiment
                    break
                else:
                    args.secondary_code = code
                    args.sentiment_code = sentiment
                    break
            while True:
                args.prompt = input("Give a generation prompt (or type q to quit): ")
                if len(args.prompt)>0:
                    break
            if args.prompt=="q":
                break

        start_len=0
        text_ids = tokenizer.encode(args.prompt)    #  encode后的编码数字,与vocab.json对应

        encoded_prompts=torch.LongTensor(text_ids).unsqueeze(0).to(args.device) # 加上device=cude：0
        args.secondary_code = " " + args.secondary_code
        args.sentiment_code = " " + args.sentiment_code
        multi_code = tokenizer.encode(args.secondary_code)  # 对主题进行编码
        senti_code = tokenizer.encode(args.sentiment_code)  # 对情感进行编码
        attr_class = 1

        # eos_token_id是结束符的编码
        generated_sequence = model.generate(
                                  input_ids=encoded_prompts,      # prompt
                                  pad_lens=None,
                                  max_length= args.length,
                                  temperature=args.temperature,
                                  top_k=args.k,
                                  top_p=args.p,
                                  repetition_penalty=args.repetition_penalty,
                                  rep_penalty_scale=args.rep_penalty_scale,
                                  eos_token_ids = tokenizer.eos_token_id,
                                  pad_token_id = 0,
                                  do_sample=args.do_sample,
                                  do_sim=args.do_sim,
                                  penalize_cond=args.penalize_cond,
                                  gedi_model=gedi_model,
                                  gpt3_api_key = args.gpt3_api_key,
                                  tokenizer=tokenizer,
                                  disc_weight=args.disc_weight,
                                  sim_weight=args.sim_weight,
                                  filter_p = args.filter_p,
                                  target_p = args.target_p,
                                  class_bias = args.class_bias,
                                  attr_class = attr_class,
                                  code_0 = args.code_undesired,
                                  code_1 = args.code_desired,
                                  multi_code=multi_code,
                                  senti_code = senti_code
                                  )

        text = tokenizer.decode(generated_sequence.tolist()[0], clean_up_tokenization_spaces=True)
        if args.gen_type == "cclm":
            text = text[start_len:]
        print("\n")
        print(text)
        print("\n")
        if not user_prompt:
            break


if __name__ == "__main__":
    main()
