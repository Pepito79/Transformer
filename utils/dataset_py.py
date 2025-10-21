
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def causal_mask(size:int):
    mask = torch.triu(input=torch.ones((size,size)), diagonal=1).type(torch.int)
    #We have an lower zero diag matrix but we want and upper one
    return mask == 0
    
    
class BillingualDataset(Dataset):
    def __init__(self,ds,src_lang,tgt_lang,max_len ,src_tokenizer,tgt_tokenizer):
        """
        ds: the dataset that we will use
        src_lang: source language
        tgt_lan: target language
        max_len : max len of the dataset 
        src_tokenizer : source tokenizer
        tgt_tokenizer: target tokenizer
        """
        super().__init__()
        self.ds = ds
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
    
        self.sos_token = torch.tensor([src_tokenizer.token_to_id("[SOS]")], dtype = torch.int64)
        self.pad_token = torch.tensor([src_tokenizer.token_to_id("[PAD]")] , dtype = torch.int64)
        self.eos_token = torch.tensor([src_tokenizer.token_to_id("[EOS]")] , dtype = torch.int64)
        
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        """
        Here are the steps to get access to the final index-th element of the dataset:
            1) Take the initial index-th element ds[index]  
            2) Split the translation attributes   
            3) Embedd the src and tgt text
            4) Add the padding for every embedding to have the same dimensions 
            5) Concat every embedding with the SOS , EOS and PAD
            6) Do the same thing with the label tensor 
        """
        src_lang = self.src_lang
        tgt_lang = self.tgt_lang
        src_tgt_pair = self.ds[index]
        src_text = src_tgt_pair['translation'][src_lang]   
        tgt_text = src_tgt_pair['translation'][tgt_lang]
        
        enc_input_tokens = self.src_tokenizer.encode(src_text).ids # [1,5,23,19] for example for one sentance : output is a list
        dec_input_tokens = self.tgt_tokenizer.encode(src_text).ids # [1,5,23,19] for example for one sentance
        
        #Every sequence has a variable len and we want to have the same length for every sentance
        enc_num_pad = self.max_len - len(enc_input_tokens) - 2 #We have the EOS and SOS tokens
        dec_num_pad = self.max_len - len(enc_input_tokens) - 1 # We only have the SOS token for the decoder
        
         
        if enc_num_pad < 0 or dec_num_pad <0 :
            raise ValueError('Sentance is too long')

        #Construct the final tensors
        encoder_input = torch.cat(
            [
               self.sos_token,
               torch.tensor(enc_input_tokens, dtype=torch.int64),
               self.eos_token,
               torch.tensor([self.pad_token] * enc_num_pad , dtype=torch.int64)
            ]
        )
        
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_pad , dtype=torch.int64)

            ]
        )
        
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_pad , dtype=torch.int64)
                
            ]
        )
        
        assert encoder_input.size(0) == self.max_len 
        assert decoder_input.size(0) == self.max_len 
        assert label.size(0) == self.max_len
        
        return {
            "encoder_input": encoder_input ,
            "decoder_input": decoder_input,
            "label":label,
            # But we will also need a mask to ignore the PAD tokens during the attention mechanism
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,max_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "src_text": src_text,
            "tgt_txt": tgt_text
        }
        