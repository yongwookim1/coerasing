from utils.arguments import TrainingArguments, DataArguments, LoraArguments

def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict) # TODO: maybe save special tokens in tokenizer
    model.resize_token_embeddings(len(tokenizer)) # this set lm_head and embed_tokens requires_grad = True

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def reshape_model_embedding(tokenizer, model):
    token_length = len(tokenizer)
    embedding_length = model.get_input_embeddings().num_embeddings
    if token_length != embedding_length:
        num_new_tokens = token_length - embedding_length
        model.resize_token_embeddings(len(tokenizer)) # this set lm_head and embed_tokens requires_grad = True
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

class BaseModel:
    def __init__(
        self,
        model_path,
        training_args: TrainingArguments,
        data_args: DataArguments,
        lora_args: LoraArguments,
        use_caption = None,
    ):
        self.model_path = model_path
        self.training_args = training_args
        self.data_args = data_args
        self.lora_args = lora_args
        self.use_caption = use_caption

        self.load_model_tokenizer()
        self.configure_special_tokens()
        self.configure_training_args()
        self.configure_peft()
        try:
            self.model.print_trainable_parameters()
        except:
            pass
        print('lljllj                        self model use_cache :', self.model.config.use_cache, flush=True)

    def configure_special_tokens(self):
        if self.use_caption and self.use_caption.get('text_pool', 'eot') == 'eot': 
            eot_token = '[EOT]'
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(additional_special_tokens=[eot_token]),
                tokenizer=self.tokenizer,
                model=self.model)
        else:
            reshape_model_embedding(self.tokenizer, self.model)
        self.model.tokenizer = self.tokenizer

    def load_model_tokenizer(self):
        raise NotImplementedError

    def configure_training_args(self):
        raise NotImplementedError
    
    def configure_peft(self):
        raise NotImplementedError

    def get_model_tokenizer(self):
        return self.model, self.tokenizer
    
    def get_model_processor(self):
        return self.model, self.processor