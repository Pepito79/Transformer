#!/usr/bin/env python
# coding: utf-8

import torch 
from torch.utils.data import random_split,DataLoader
import import_ipynb
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from dataset_py import BillingualDataset,causal_mask
from model import build_transformer
from config import get_config , get_weights_file_path
from  tqdm import tqdm
import warnings
from model import Transformer


def greedy_decoder(model: Transformer ,source,source_mask, tgt_tokenizer : Tokenizer,max_len: int,device):
    """
    model : Transformer
    source_mask : tensor
    tgt_tokenizer : Tokenizer
    max_len : int
    
    
    Here is an example:
    "Hello man"   ---> [token(SOS), token(Hello), token(my), token(friend)]    #output_size = (len(output))
    """

    sos_id = tgt_tokenizer.token_to_id("[SOS]")
    eos_token = tgt_tokenizer.token_to_id("[EOS]")
    
    #Use the encoder to have a bette represenation of words (we compute it once)
    encoder_output = model.encode(source,source_mask)
    #Give to the deocder the first token which is the SOS token
    decoder_input = torch.empty(1,1).type_as(source).fill_(sos_id).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(decoder_input,encoder_output,source_mask,decoder_mask)
        #Project to the vocab dimension 
        prob = model.project(out[:,-1])
        _,next_token = torch.max(prob,dim=1)
        decoder_input = torch.concat([
            decoder_input,
            torch.empty(1,1).type_as(source).fill_(next_token.item())
        ], dim=1)
        
        if next_token == eos_token:
            break
        
        #We return the finale sentance without the batch dimension (which is 1)
    return decoder_input.squeeze(0)
        
def run_validation(model: Transformer,validation_ds,tokenizer_src:Tokenizer,tokenizer_tgt: Tokenizer,device,print_msg, max_len:int,num_examples =2):
    """
    model  : the model 
    validation_ds : validation dataset , here it's the billangual class
    tokenizer_tgt:  target tokenizer
    tokenizer_src : source tokenizer
    device : "cuda" or "cpu"
    print_msg: printer
    max_len: max len of tokens in one sentance
    num_examples : the number of examples 
    """
    
    #Put the model in the validation state by disabling the gradient 
    model.eval()
    count = 0 # monitor the number of examples evaluated
    var = 80  # variable to espace the output
    
    for batch in validation_ds:
        count +=1
        # Take the encoder and decoder inputs from the batc
        encoder_input = batch["encoder_input"].to(device)
        encoder_mask = batch["encoder_mask"].to(device)
        
        assert encoder_input.size(0) == 1 , "Batch size must be 1"
        model_out = greedy_decoder(model,encoder_input,encoder_mask,tokenizer_tgt,max_len,device)  
        
        source_text = batch["src_text"][0]  # Take [0] because src_text = ["blablabla"]
        tgt_text= batch['tgt_txt'][0]
        #Transform the tokens in text , we need to make it a numpy 
        predicted = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
        
        print_msg("-"*var)
        print_msg(f'SOURCE : {source_text}')
        print_msg(f'TARGET : {tgt_text}')
        print_msg(f'PREDICTED: {predicted}')
        
        if count == num_examples:
            break

def get_sentances(ds,lang):
    for item in ds:
        yield item["translation"][lang]

def get_or_build_tokenizer(config,ds , lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(model= WordLevel(unk_token= "[UNK]"))
        tokenizer.pre_tokenizer = Whitespace() 
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]" , "[SOS]","[EOS]"],min_frequency=2)
        tokenizer.train_from_iterator(iterator=get_sentances(ds,lang),trainer=trainer)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):

    lang_src = config["lang_src"]
    lang_tgt = config["lang_tgt"]
    len_seq = config["seq_len"]

    #Load the dataset from huggingface
    ds_raw = load_dataset("opus_books",f"{lang_src}-{lang_tgt}", split="train")

    #We define two tokenizer (each language has different tokens)
    tokenizer_src = get_or_build_tokenizer(config,ds_raw,lang_src)
    tokenizer_tgt = get_or_build_tokenizer(config,ds_raw,lang_tgt)

    #Define the size of the trainning and validation sets
    train_size = int(0.9 * len(ds_raw))
    val_size = len(ds_raw) - train_size
    train_ds_raw , val_ds_raw = random_split(ds_raw, [train_size,val_size])

    train_ds = BillingualDataset(ds =train_ds_raw,src_lang= lang_src , tgt_lang= lang_tgt,src_tokenizer= tokenizer_src, tgt_tokenizer=tokenizer_tgt , max_len=len_seq)
    val_ds = BillingualDataset(ds =val_ds_raw,src_lang= lang_src , tgt_lang= lang_tgt,src_tokenizer= tokenizer_src, tgt_tokenizer=tokenizer_tgt , max_len=len_seq)

    #Find the max len sentance 
    max_len_src = 0
    max_len_tgt = 0 
    for item in ds_raw:
        src_idc = tokenizer_src.encode(item["translation"][lang_src]).ids
        tgt_idc = tokenizer_tgt.encode(item["translation"][lang_tgt]).ids

        max_len_src = max(max_len_src, len(src_idc))
        max_len_tgt= max(max_len_tgt,len(tgt_idc))

        print(f'Max len source : {max_len_src}')
        print(f'Max len tgt : {max_len_tgt}')

    #Load the final datasets
    train_dataloader = DataLoader(dataset=train_ds,batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(dataset=val_ds,batch_size=1, shuffle=True)

    return train_dataloader , val_dataloader , tokenizer_src , tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def train_model(config):

    #Define in which device the training will be
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(f" Using device:{device}")

    #Create the folder to save the model
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    #Load the dataset
    train_dataloader , val_dataloader , tokenizer_src , tokenizer_tgt = get_ds(config=config)

    #Import the model
    model = get_model(config= config,vocab_src_len= tokenizer_src.get_vocab_size(), vocab_tgt_len= tokenizer_tgt.get_vocab_size()).to(device)

    # Start tensoboard to visualize the loss charts
    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters() , lr=config['lr'], eps= 1e-9)

    initial_epoch = 0
    global_step = 0

    if config["preload"]:
        model_filename = get_weights_file_path(config=config,epoch=config["preload"])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state["global_step"]

    loss_fn= nn.CrossEntropyLoss(ignore_index= tokenizer_src.token_to_id("[PAD]"),label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch,config["num_epochs"]):

        batch_iterator = tqdm(train_dataloader, desc = f'Processing epoch {epoch}')
        for batch in batch_iterator:
            
            #Free the unsed cache created by CUDA
            torch.cuda.empty_cache()
            model.train()

            encoder_input = batch["encoder_input"].to(device) #(B , seq_len)
            decoder_input = batch["decoder_input"].to(device) #(B , seq_len)

            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(decoder_input,encoder_output, encoder_mask,decoder_mask)
            proj_output = model.project(decoder_output)
            label = batch["label"].to(device)

            print("proj_output.shape:", proj_output.shape)
            print("label.shape:", label.shape)


            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            #Write the loss in tensorboard
            writer.add_scalar("train_loss", loss.item(), global_step=global_step)
            writer.flush()
            
            # Backpropagate the loss
            loss.backward()
            
            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
        
        
        run_validation(model=model,validation_ds=val_dataloader,tokenizer_src=tokenizer_src,tokenizer_tgt=tokenizer_tgt,device=device,print_msg=lambda msg: batch_iterator.write(msg),max_len=config['seq_len'])      

        
        #Save at every epoch the model 
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)