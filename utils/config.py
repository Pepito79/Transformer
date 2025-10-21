from pathlib import Path
def get_config():
    return {
        "batch_size":5,
        "num_epochs":10,
        "lr":0.00001,
        "seq_len":350 ,  
        "d_model":512,
        "lang_src": "fr",
        "lang_tgt": "it",
        "model_folder":"weights",
        "model_basename":"tmodel_",
        "preload": None,
        "tokenizer_file":"tokenizer_{0}.json",
        "experiment_name":"runs/tmodel"
    }
    

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config["model_basename"]
    model_filename= f'{model_basename}{epoch}.pt'
    return Path(".")/model_folder/model_filename
    