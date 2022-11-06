
import argparse
import glob
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import sys
from dataset import tokenize, create_dataset

from sklearn.metrics import matthews_corrcoef, f1_score
from discriminator_seqGAN import Discriminator
from modeling_gpt2 import GPT2LMHeadModel

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    FlaubertConfig,
    FlaubertForSequenceClassification,
    FlaubertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
    GPT2Config,
    GPT2Tokenizer,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors


# https://github.com/huggingface/transformers/blob/master/src/transformers/data/metrics/__init__.py
def acc_and_f1(preds, labels):
    assert len(preds) == len(labels)
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def simple_accuracy(preds, labels):
        return (preds == labels).mean()


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)
ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            XLNetConfig,
            XLMConfig,
            RobertaConfig,
            DistilBertConfig,
            AlbertConfig,
            XLMRobertaConfig,
            FlaubertConfig,
            GPT2Config
        )
    ),
    (),
)
MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
}


#GPT2 added as per - https://huggingface.co/transformers/model_doc/gpt2.html
def add_sep(batch, sep_id):
    batch[0]
    len_list = (batch[1].sum(dim=1) - batch[2].sum(dim=1)).tolist()
    left_chunk = [x[:len_] for x,len_ in zip(batch[0],len_list)]
    right_chunk= [x[len_:] for x,len_ in zip(batch[0],len_list)]
    mid_chunk = [torch.Tensor(sep_id).type_as(x) for x in batch[0]]
    tensor_list = [torch.cat((left,mid,right)) for (left,mid,right) in
                   zip(left_chunk, mid_chunk, right_chunk)]
    return torch.stack(tensor_list)[:,:-1]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train_discriminator(discriminator, optimizer, source_dataset, target_dataset, d_steps, epochs):
    source_sampler = RandomSampler(source_dataset)
    source_dataloader = DataLoader(source_dataset, sampler=source_sampler, batch_size=20)

    target_sampler = RandomSampler(target_dataset)
    target_dataloader = DataLoader(target_dataset, sampler=target_sampler, batch_size=20)

    for d in range(d_steps):
        # i = 0
        for step, batch in enumerate(zip(source_dataloader, target_dataloader)):
            # device = torch.device("cuda")
            pos_samples = batch[0][0]
            pos_targets = batch[0][1]
            neg_samples = batch[1][0]
            neg_targets = batch[1][1]
            inp = torch.cat((pos_samples, neg_samples), 0).type(torch.LongTensor) #[b_size1+b_size2, 1024]
            target = torch.cat((pos_targets, neg_targets), 0).type(torch.LongTensor) #[b_size1+b_size2, 1]
            # i = i+1
            # print("i:",i)
            # print(inp.shape, target.shape)
            # shuffle
            perm = torch.randperm(target.size()[0])
            target = target[perm]
            inp = inp[perm]

            inp = Variable(inp).cuda()
            target = Variable(target).cuda()

            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            optimizer.zero_grad()
            discriminator = discriminator.cuda()
            out = discriminator.batchClassify(inp)
            loss_fn = nn.BCEWithLogitsLoss()
            # target = target.squeeze(1)
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            out = out.unsqueeze(1)  # (2*batch_size,1)
            target = target.float()
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
            # exit()
            total_loss += loss.data.item()
            total_acc += torch.sum((out>0.5)==(target>0.5)).data.item()
            if step % 10 == 0 :
                sys.stdout.flush()
        total_loss /= step
        total_acc /= 110


def discriminator_for_gen(discriminator, optimizer, inp):
    out = discriminator.batchClassify(inp)
    target = torch.ones(1).cuda()
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(out, target)
    loss.backward()     ##
    optimizer.step()
    return loss


def train(args, train_dataset, source_dataset, model, discriminator, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # randomsampler 返回的是可迭代对象  ，默认不重复采样
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    if args.max_steps > 0:      # -1
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    logger.info("  len(train_dataloader) = %d", len(train_dataloader))  # # 一个epoch的话是30000 train_dataset为240000
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    # 可以设置分层学习率，分层权重衰减
    optimizer_grouped_parameters = [
        {   # n是模型每层名字，p是每层参数，any判断可迭代对象是否全为空、0、false。是则返回false否则返回true
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        logger.info("  load saved optimizer and scheduler")
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        torch.cuda.empty_cache()    # 自动回收不用的显存
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False,
        )
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))      # len(train_dataset)==240000
    logger.info("  Num Epochs = %d", args.num_train_epochs)     # Num Epochs==1
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)     # 1
    logger.info("  Total optimization steps = %d", t_total)     # 30000
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()   # 模型中参数的梯度设为0
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],)
    set_seed(args)  # Added here for reproductibility
    pt_id = tokenizer.encode(args.code_0)[0]    # args.code_0 : false.9562
    nt_id = tokenizer.encode(args.code_1)[0]    # # args.code_1 : true.7942
    if args.threeway:
        nl_id = tokenizer.encode('neutral')[0]
    logger.info("  begin training")
    flag = 1
    for epoch_ in train_iterator:
        import gc
        gc.collect()
        epoch_iterator = train_dataloader   # 30000个数据
        for step, batch in enumerate(epoch_iterator):   # enumerate将数据对象的索引和数据返回，step为索引，batch为数据
            torch.cuda.empty_cache()
            if args.sst5:
                split_lookup = {0:0, 1:1, 2:2, 3:3, 4:3, 5:3}   #epoch:threshold_to_split
                threshold = split_lookup[epoch_]
                new_labels = ( batch[3] > threshold ).type_as(batch[3])
                batch[3] = new_labels
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:      # 从上次的训练结束初开始训练
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)     # 创建元组
            if args.model_type == 'gpt2':
                # batch是3*192，第一个是句子的encode，第二个是用1代表单词，第三个全是0，三个都是4*192
                # batch_0是只有batch的第一部分，4*192，用encode表示的
                batch_0 = add_sep(batch, tokenizer.encode('<sep>')) if args.add_sep else batch[0]   # add_sep=false
                # if flag == 1 :
                #     flag = 0
                #     logger.info("  batch = %s", batch)

                # prepending tokens corresponding to 'positive' and 'negative' to the inputs
                seq_a = (torch.ones(batch_0.shape[0])*pt_id).type_as(batch_0).view(-1,1)    #
                seq_b = (torch.ones(batch_0.shape[0])*nt_id).type_as(batch_0).view(-1,1)
                if args.threeway:
                    seq_c = (torch.ones(batch_0.shape[0])*nl_id).type_as(batch_0).view(-1,1)
                seq_a = torch.cat((seq_a, batch_0), dim=1)[:,:-1]   # 输入句子的整数encode，seq_a.shape=4*192，4是batchsize，192是句子长度，句子不够长的，用50256代替
                seq_b = torch.cat((seq_b, batch_0), dim=1)[:,:-1]   # dim=1表示按维数1拼接（横着拼），即将pt_id或者nt_id放到首位
                bsz = seq_a.shape[0]    # 4
                seq_batched = torch.cat((seq_a,seq_b),dim=0)    # seq_batched.shape=8*192,将正负的batch连接起来
                # want to compute LM loss here so feeding inputs as labels
                inputs = {"input_ids": seq_batched, "attention_mask": None, "labels": seq_batched}
            else:
                assert args.model_type == 'gpt2' #let's not support other models for now
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            # inputs的iput_ids和labels都是8*192，前面4*192带上积极标签，后面4*192带上消极标签
            outputs = model(**inputs)   # modeling_gpt2.py modified to have none reduction 预测结果
            losses = outputs[0].view(seq_batched.shape[0], -1)
            # loss mask includes first padded token  数据增强用mask
            if args.mask_eos_token:     # false
                loss_mask = batch[1][:,:-1].to(torch.float16).cuda()
                if args.add_sep:
                    raise NotImplementedError
            else:
                loss_mask = batch[1][:,:-1].to(torch.float32).cuda()
                #appending with ones to account for the control code token being added
                if args.add_sep:    # false
                    #adding the sep token would require extending the loss mask of ones by one position to the right (equivalent to prepending one on the left)
                    left_ = torch.ones(loss_mask.shape[0],2).type_as(loss_mask)
                    loss_mask = torch.cat((left_, loss_mask[:,:-2]), dim=1)
                else:
                    left_ = torch.ones(loss_mask.shape[0],1).type_as(loss_mask)
                    loss_mask = torch.cat((left_, loss_mask[:,:-1]), dim=1)
            loss_lengths = torch.sum(loss_mask,1,keepdim=True)  # 4个loss
            # losses大小为8*191,loss_a大小为4*191
            loss_a,loss_b=torch.split(losses, bsz, dim=0)       # 第二个参数是分割的块数，dim是分割的维度，但实际在这里只分割了两块，因为返回的接收变量只有两个
            loss_a*=loss_mask
            loss_b*=loss_mask

            gen_loss_a = (batch[3]==0).to(torch.float32).unsqueeze(1)*loss_a/loss_lengths
            gen_loss_b = (batch[3]==1).to(torch.float32).unsqueeze(1)*loss_b/loss_lengths
            if args.jigsaw and args.jigsaw_no_toxic_gen:
                gen_loss = torch.sum(gen_loss_a+gen_loss_b)/bsz
            else:
                gen_loss = torch.sum(gen_loss_a+gen_loss_b)/bsz
            if args.sum_loss:
                loss_a = loss_a.sum(dim=1)
                loss_b = loss_b.sum(dim=1)
            else:
                loss_a = (loss_a/loss_lengths).sum(dim=1)
                loss_b = (loss_b/loss_lengths).sum(dim=1)

            class_logits = torch.stack((-loss_a, -loss_b), dim=1)   #(bsz, 2) dimensional
            batch[3][batch[3] == 2] = 1     #turning 3-ary to binary
            class_labels = batch[3]     # [1,0,1,0],4个label
            if args.logit_scale:    # true
                if args.fp16:
                    if not isinstance(model,torch.nn.DataParallel) and not isinstance(model,torch.nn.parallel.DistributedDataParallel):
                        class_logits*=model.logit_scale.float()
                    else:
                        class_logits*=model.module.logit_scale.float()
                else:
                    if not isinstance(model,torch.nn.DataParallel) and not isinstance(model,torch.nn.parallel.DistributedDataParallel):
                        class_logits*=model.logit_scale
                    else:
                        class_logits*=model.module.logit_scale
            if args.do_adversarial == True  and (step + 1) % 100 == 0 and step < 10000:
                # model_for_gen = model.module
                model_for_gen = model.to(args.device)
                text_ids = tokenizer.encode(args.prompt)
                encoded_prompts = torch.LongTensor(text_ids).unsqueeze(0).to(args.device)
                multi_code = tokenizer.encode(args.control_code)
                senti_code = tokenizer.encode(args.senti_code)
                # print('prompt, code: ', args.prompt, args.control_code)
                gen_file = open(args.output_dir+'generated_sentence.txt', 'a')
                for i in range(5):  ##if generate more...it will take training so much time :(
                    generated_sequence = model_for_gen.generate(
                        input_ids=encoded_prompts,
                        pad_lens=None,
                        max_length=192,
                        temperature=0.1,
                        top_k=50,
                        top_p=1,
                        repetition_penalty=1.2,
                        rep_penalty_scale=10,
                        eos_token_ids=tokenizer.eos_token_id,
                        pad_token_id=0,
                        do_sample=True,
                        do_sim=True,
                        penalize_cond=True,
                        gedi_model=model_for_gen,
                        gpt3_api_key=None,
                        tokenizer=tokenizer,
                        disc_weight=40,
                        sim_weight=3,
                        filter_p=0.8,
                        target_p=0.8,
                        class_bias=args.class_bias,
                        attr_class=1,
                        code_0=args.code_undesired,
                        code_1=args.code_desired,
                        multi_code=multi_code,
                        senti_code = senti_code
                    )
                    generated_text = tokenizer.decode(generated_sequence.tolist()[0], clean_up_tokenization_spaces=True)
                    # print('generated_text: ', generated_text)
                    gen_file.write(generated_text)
                    gen_file.write("\n")
                    # print('generated_sequence: ', generated_sequence, generated_sequence.shape)
                gen_file.close()
            elif args.do_adversarial == True and step == 10000:
                target_dataset = create_dataset(args.output_dir+'generated_sentence.txt', tokenizer, 0)
                optimizer_dis = optim.AdamW(discriminator.parameters(),lr = 2e-4, eps=args.adam_epsilon)
                train_discriminator(discriminator, optimizer_dis, source_dataset, target_dataset, d_steps=20, epochs=3)
                # D_loss = maximum_mean_discrepancy(,)
                # tr_loss += D_loss.item()
            elif args.do_adversarial == True and (step + 1) % 10 == 0 and step > 10000:
                # model_for_gen = model.module
                model_for_gen = model.to(args.device)
                text_ids = tokenizer.encode(args.prompt)
                encoded_prompts = torch.LongTensor(text_ids).unsqueeze(0).to(args.device)

                multi_code = tokenizer.encode(args.control_code)
                senti_code = tokenizer.encode(args.senti_code)
                generated_sequence = model_for_gen.generate(
                    input_ids=encoded_prompts,
                    pad_lens=None,
                    max_length=192,
                    temperature=0.1,
                    top_k=50,
                    top_p=1,
                    repetition_penalty=1.2,
                    rep_penalty_scale=10,
                    eos_token_ids=tokenizer.eos_token_id,
                    pad_token_id=0,
                    do_sample=True,
                    do_sim=True,
                    sim_weight=3,
                    penalize_cond=True,
                    gedi_model=model_for_gen,
                    gpt3_api_key=None,
                    tokenizer=tokenizer,
                    disc_weight=40,
                    filter_p=0.8,
                    target_p=0.8,
                    class_bias=args.class_bias,
                    attr_class=1,
                    code_0=args.code_undesired,
                    code_1=args.code_desired,
                    multi_code=multi_code,
                    senti_code=senti_code
                )
                D_loss = discriminator_for_gen(discriminator, optimizer_dis, generated_sequence)
                with open(args.output_dir+'D_loss.txt', "a") as data_json:
                    data_json.write(f'step = {step} \t D_loss is = {str(D_loss)}\n')
                # if n_steps > 1:
                #     D_loss = D_loss / n_steps
                # D_loss.backward()
                tr_loss += D_loss.item()
                print("step:",step+1)

            loss_fn = torch.nn.CrossEntropyLoss()
            # 损失函数
            loss = loss_fn(class_logits, class_labels)*args.disc_weight + args.gen_weight*gen_loss
            if np.isnan(loss.detach().cpu().numpy()):   # 测试元素是否为NaN
                import pdb; pdb.set_trace()
                print("404_nan:")
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:    # 1
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:   # true
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            tr_loss += loss.item()  # item转换为浮点数    tr_loss为训练集loss
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    # evaluate_during_training=false
                    if (args.local_rank == -1 and args.evaluate_during_training):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value
                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss
                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))
                    with open(args.output_dir + 'str_loss.txt', "a") as data_json:
                        data_json.write(f'step = {step} \t D_loss is = {str(tr_loss)}\n')
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:     # args.max_steps=-1
            train_iterator.close()
            break
    if args.local_rank in [-1, 0]:
        tb_writer.close()
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)
    results = {}
    # zip将一系列可迭代的对象打包成元组，然后组成列表，返回
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel) and not isinstance(model,torch.nn.parallel.DistributedDataParallel):
            model = torch.nn.DataParallel(model)
        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))       # 15200
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        overall_gen_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                batch_0 = add_sep(batch, tokenizer.encode('<sep>')) if args.add_sep else batch[0]
                #this is assuming you get only one token for the words 'positive' and 'negative'
                pt_id = tokenizer.encode(args.code_0)[0]    #pt_id = tokenizer.encode('positive')[0]
                nt_id = tokenizer.encode(args.code_1)[0]    #nt_id = tokenizer.encode('negative')[0]
                if args.threeway:
                    nl_id = tokenizer.encode('neutral')[0]
                #prepending tokens corresponding to 'positive' and 'negative' to the inputs
                seq_a = (torch.ones(batch_0.shape[0])*pt_id).type_as(batch_0).view(-1,1)
                seq_b = (torch.ones(batch_0.shape[0])*nt_id).type_as(batch_0).view(-1,1)
                if args.threeway:
                    seq_c = (torch.ones(batch_0.shape[0])*nl_id).type_as(batch_0).view(-1,1)
                seq_a = torch.cat((seq_a, batch_0), dim=1)[:,:-1]
                seq_b = torch.cat((seq_b, batch_0), dim=1)[:,:-1]
                bsz = seq_a.shape[0]
                if args.threeway:
                    seq_c = torch.cat((seq_c, batch_0), dim=1)[:,:-1]
                    seq_batched = torch.cat((seq_a,seq_b,seq_c),dim=0)
                else:
                    seq_batched = torch.cat((seq_a,seq_b),dim=0)
                #want to compute LM loss here so feeding inputs as labels
                inputs = {"input_ids": seq_batched, "attention_mask": None, "labels": seq_batched}
                outputs = model(**inputs) #modeling_gpt2.py changed to have none reduction
                losses = outputs[0].view(seq_batched.shape[0], -1)
                #loss mask includes first padded token
                if args.mask_eos_token:
                    loss_mask = batch[1][:,:-1].to(torch.float16).cuda()
                else:
                    loss_mask = batch[1][:,:-1].to(torch.float32).cuda()
                    #appending with ones to account for the control code token being added
                    left_ = torch.ones(loss_mask.shape[0],1).type_as(loss_mask)
                    loss_mask = torch.cat((left_, loss_mask[:,:-1]), dim=1)
                loss_lengths = torch.sum(loss_mask,1,keepdim=True)
                loss_a,loss_b=torch.split(losses, bsz, dim=0)
                loss_a*=loss_mask
                loss_b*=loss_mask
                #weights all batches equally even if sum of seqlens vary accross training iterations
                if False:
                    gen_loss_a = (batch[3]==0).to(torch.float16).unsqueeze(1)*loss_a/loss_lengths
                    gen_loss_b = (batch[3]==1).to(torch.float16).unsqueeze(1)*loss_b/loss_lengths
                else:
                    gen_loss_a = (batch[3]==0).to(torch.float32).unsqueeze(1)*loss_a/loss_lengths
                    gen_loss_b = (batch[3]==1).to(torch.float32).unsqueeze(1)*loss_b/loss_lengths
                gen_loss = torch.sum(gen_loss_a+gen_loss_b)/bsz
                if args.sum_loss:
                    loss_a = loss_a.sum(dim=1)
                    loss_b= loss_b.sum(dim=1)
                else:
                    loss_a = (loss_a/loss_lengths).sum(dim=1)
                    loss_b= (loss_b/loss_lengths).sum(dim=1)
                class_logits = torch.stack((-loss_a, -loss_b), dim=1) #(bsz, 2) dimensional
                batch[3][batch[3] == 2] = 1  #turning 3-ary to binary
                class_labels = batch[3]
                loss_fn = torch.nn.CrossEntropyLoss()
                if args.logit_scale:
                    if args.fp16:
                        if not isinstance(model,torch.nn.DataParallel) and not isinstance(model,torch.nn.parallel.DistributedDataParallel):
                            class_logits*=model.logit_scale.float()
                        else:
                            class_logits*=model.module.logit_scale.float()
                    else:
                        if not isinstance(model,torch.nn.DataParallel) and not isinstance(model,torch.nn.parallel.DistributedDataParallel):
                            class_logits*=model.logit_scale
                        else:
                            class_logits*=model.module.logit_scale.float()
                if args.outbias:
                    if args.fp16:
                        class_logits+=model.bias.float()
                    else:
                        class_logits+=model.bias
                loss = loss_fn(class_logits, class_labels)      # 损失函数
                tmp_eval_loss = loss
                tmp_gen_loss = gen_loss
                logits = class_logits
                eval_loss += tmp_eval_loss.mean().item()
                overall_gen_loss += tmp_gen_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = class_labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, class_labels.detach().cpu().numpy(), axis=0)
            # preds是2*8的预测，浮点数
            # out_label_ids，batchsize是8，所以8个label，0，1表示
        eval_loss = eval_loss / nb_eval_steps
        overall_gen_loss = overall_gen_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)    # 取出数组中每行的最大值，构成一维数组
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)       # shape中为1的维度去掉
        # preds为0，1表示的预测
        # out_label_ids为label
        # len(out_label_ids)为15200
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)
        #avg gen_loss over all val samples
        #will be used to compute perplexity for the conditional seq do_lower_case
        result.update({'overall_gen_loss':overall_gen_loss})
        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            if eval_task == 'cola':
                accf1 = acc_and_f1(preds, out_label_ids)
                print(accf1)
                for key in sorted(accf1.keys()):
                    logger.info("  %s = %s", key, str(accf1[key]))
                    writer.write("%s = %s\n" % (key, str(accf1[key])))
    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:   # overwrite_cache=true
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        # 获得训练数据集或者测试数据集
        examples = (processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir))

        if args.model_type == 'gpt2': # setting pad token for GPT-2
            tokenizer.pad_token = '[PAD]'
        if args.sst5:
            label_list = ['0','1','2','3','4']
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,        # 分类还是回归
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)      # cached_features存放模型用到的数据
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Convert to Tensors and build dataset
    logger.info("len( features ) %s", len(features))
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    logger.info("end load_and_cache_examples")
    return dataset


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_dir",default=None,type=str,required=True,help="The input data dir. Should contain the .tsv files (or other data files) for the task.",)
    parser.add_argument("--model_type",default=None,type=str,required=True,help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),)
    parser.add_argument("--discriminator", default=None, type=str, required=True, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),)
    parser.add_argument("--model_name_or_path",default=None,type=str,required=True,help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),)
    parser.add_argument("--task_name", default=None, type=str, required=True, help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),)
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.",)

    #new generative classifier specific parameters
    parser.add_argument("--dropout",default=0.1,type=float, help="dropout prob")
    parser.add_argument("--gen_weight",default=0.0,type=float, help="scalar multiple for generative loss (lambda)")
    parser.add_argument("--logit_scale",action="store_true",help="learns to scale logits for classification")
    parser.add_argument("--threeway", action="store_true", help="does 3-way classification")
    parser.add_argument("--sum_loss",action="store_true", help="sums losses")
    parser.add_argument("--outbias",action="store_true", help="learns output bias for each class")

    # Other parameters
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",)
    parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name",)
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3",)
    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",)
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    parser.add_argument("--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",)
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",)
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",)
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",)
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")

    parser.add_argument("--eval_all_checkpoints", action="store_true", help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",)
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",)
    parser.add_argument("--do_adversarial", action="store_true", help='adversarial training')

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
    parser.add_argument("--fp16_opt_level", type=str, default="O1", help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].See details at https://nvidia.github.io/apex/amp.html",)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--mask_eos_token", action="store_true", help="whether to mask eos token loss or not; prefer masking if training for DA",)
    parser.add_argument("--add_sep", action="store_true", help="Include sep token if this arg is used between the two sentences in a pair | can/should be used for mrpc/mnli/qqp/qnli")
    parser.add_argument("--sst5", action="store_true", help="custom ops for SST-5")

    parser.add_argument("--jigsaw", action="store_true", help="custom setup for jigsaw")
    parser.add_argument("--jigsaw_no_toxic_gen", action="store_true", help="custom setup for jigsaw - gen_loss used only for non-toxic samples | check training loop")
    parser.add_argument("--code_0", type=str, default="negative", help="control code to be used for code 1 of 2 (we support 3 at most - with the third one = 'neutral' for now)")
    parser.add_argument("--code_1", type=str, default="positive", help="control code to be used for code 2 of 2 (we support 3 at most - with the third one = 'neutral' for now)")
    parser.add_argument("--senti_code", type=str, default="positive", help="positive or negative")
    parser.add_argument("--prompt", type=str, default="It is say that", help="Initially given text")
    parser.add_argument("--control_code", type=str, default="movie", help="control code")
    parser.add_argument("--class_bias", type=float, default=None,help="biases GeDi's classification probabilties")
    parser.add_argument("--sim_weight", type=float, default=5.0,help="weight for GeDi discriminator",)
    parser.add_argument("--do_sim", action="store_true",help="If set to False no calulate similarity. Otherwise calulate similarity is used. Defaults to True.")


#     args = parser.parse_args()
    args, unknown = parser.parse_known_args()   # 命令行解析
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)     # 返回指定的文件夹包含的文件或文件夹的名字的列表。
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:   # 不进行分布式学习
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    args.code_desired = "true"
    args.code_undesired = "false"
    args.disc_weight = 1.0 - args.gen_weight
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-[line:%(lineno)d]-   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    if args.class_bias is None:
        args.class_bias = 0.0
    # Set seed
    set_seed(args)
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()     # sst-2是二分类任务，label_list==['0','1']
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    args.model_type = args.model_type.lower()   # gpt-2
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,  # config_name为空
        num_labels=num_labels,  # num_labels==2
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.outbias:    # false
        if args.threeway:
            config.nbias=3
        else:
            config.nbias=2
    else:
        config.nbias=0
    config.embd_pdrop = args.dropout
    config.attn_pdrop = args.dropout
    config.resid_pdrop = args.dropout
    if args.logit_scale:    # true
        config.logit_scale=True
    else:
        config.logit_scale=False
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,    # tokenizer_name为空
        do_lower_case=args.do_lower_case,   # false
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.add_sep:    #false
        special_tokens_dict = {'sep_token': '<SEP>'}
        tokenizer.add_special_tokens(special_tokens_dict)
    config.output_past = True   # https://github.com/huggingface/transformers/pull/3734
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,   # cache_dir=none
    )
    model.resize_token_embeddings(len(tokenizer))
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    model.to(args.device)

    source_file = '/root/nlp/Gedi_test/data/TF_all/movie_true_positive_500.txt'
    source_dataset = create_dataset(source_file, tokenizer, label=1)
    discriminator = Discriminator(embedding_dim=64, hidden_dim=64, vocab_size=50257, max_seq_len=192, gpu=True)
    # discriminator = discriminator.to(args.device)
    args.control_code = " " + args.control_code
    args.senti_code = " " + args.senti_code
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset,source_dataset, model,discriminator, tokenizer)
        logger.info("end training")
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        # Load a trained model and vocabulary that you have fine-tuned

        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)
    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        logger.info("begin eval")
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:   # false
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            logger.info("global_step:%s",global_step)
            logger.info("prefix:%s",prefix)
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)
    return results


if __name__ == "__main__":
    main()
