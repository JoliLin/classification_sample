#copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

    
import copy
import math
import os
import json
import tarfile
from .file_utils import cached_path

import torch
from torch import nn

def create_saving_dir(save):
    if not save.endswith('/'):
        save = save + '/'
    try:
        os.mkdir(save)
    except FileExistsError:
        print(f'Dir : {save} existed.')

class BertConfig(object):
    def __init__(self, vocab_size, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act="gelu", hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=16, initializer_range=0.02, extra_dim=None, hidden_size_aug=204, adapter_size=64, trainable_layer=3):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.adapter_size = adapter_size
        self.trainable_layer = trainable_layer

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

PRETRAINED_MODEL_ARCHIVE_MAP = {'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz", 'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz", 'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",}

class PreTrainedBertModel(nn.Module):
    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        self.config = config

    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.beta.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.gamma.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, cache_dir=None, *inputs, **kwargs):
        if pretrained_model_name in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name]
        else:
            archive_file = pretrained_model_name
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except FileNotFoundError:
            print('FileNotFound')

            return None
        #print(resolved_archive_file)
        
        if resolved_archive_file == archive_file:
            print('load file')
        else:
            print(f'from {resolved_archive_file}')

        #tempdir = None
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            #tempdir = tempfile.mkdtemp()
            create_saving_dir('../'+pretrained_model_name) 
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall('../'+pretrained_model_name)
            serialization_dir = '../'+pretrained_model_name
        
        print(serialization_dir)
        config_file = os.path.join(serialization_dir, 'bert_config.json')
        config = BertConfig.from_json_file(config_file)
        model = cls(config, *inputs, **kwargs)
        weights_path = os.path.join(serialization_dir, 'pytorch_model.bin')
        state_dict = torch.load(weights_path)
        
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata
        
        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            #print(unexpected_keys)

            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix+name+'.')
        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        '''
        if tempdir:
            shutil.rmtree(tempdir)
        '''
        
        return model
#'''
class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x-u).pow(2).mean(-1, keepdim=True)
        x = (x-u) / torch.sqrt(s+self.variance_epsilon)
        return self.gamma*x+self.beta
#'''

#BertLayerNorm = torch.nn.LayerNorm

class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings,self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        '''
        input_ids: [batch_size, seq_len]
        token_type_ids: [batch_size, seq_len]
        '''

        word_embeddings = self.word_embeddings(input_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = word_embeddings+position_embeddings+token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class BertLayer(nn.Module):
    def __init__(self, config, layer_num):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config, layer_num)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config, layer_num)
        

    def forward(self, hidden_states, attention_mask, attention_show_flg=False):
        if attention_show_flg == True:
            attention_output, attention_probs = self.attention(hidden_states, attention_mask, attention_show_flg)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
            return layer_output, attention_probs

        elif attention_show_flg == False:
            attention_output = self.attention(hidden_states, attention_mask, attention_show_flg)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
            return layer_output

class BertAttention(nn.Module):
    def __init__(self, config, layer_num):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config, layer_num)

    def forward(self, input_tensor, attention_mask, attention_show_flg=False):
        if attention_show_flg == True:
            self_output, attention_probs = self.self(input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)
            return attention_output, attention_probs

        elif attention_show_flg == False:
            self_output = self.self(input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)
            return attention_output

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size/config.num_attention_heads)
        self.all_head_size = self.num_attention_heads*self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
    
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        '''
        batch_size, seq_len, hidden
        => batch_size, 12, seq_len, hidden/12
        '''
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3 )

    def forward(self, hidden_states, attention_mask, attention_show_flg=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2]+ (self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)

        if attention_show_flg == True:
            return context_layer, attention_probs
        elif attention_show_flg == False:
            return context_layer

class Adapter(nn.Module):
    def __init__(self, hidden_size, adapter_size):
        super(Adapter, self).__init__()
        self.encoder = nn.Linear(hidden_size, adapter_size)
        self.decoder = nn.Linear(adapter_size, hidden_size)

    def forward(self, hidden_states):
        adapter_hidden_states = self.encoder(hidden_states)
        adapter_hidden_states = gelu(adapter_hidden_states)
        adapter_hidden_states = self.decoder(adapter_hidden_states)
        return adapter_hidden_states + hidden_states

class BertSelfOutput(nn.Module):
    def __init__(self, config, layer_num):
        super(BertSelfOutput, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        ###
        self.adapter = Adapter(hidden_size=config.hidden_size, adapter_size=config.adapter_size)
        ###
        self.layer_num = layer_num
        self.trainable_layer = config.trainable_layer

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.layer_num > self.trainable_layer:
            hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states+input_tensor)
        return hidden_states 
   
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    #return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config, layer_num):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        ###
        self.adapter = Adapter(hidden_size=config.hidden_size, adapter_size=config.adapter_size)
        ###
        self.layer_num = layer_num
        self.trainable_layer = config.trainable_layer

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        ###
        if self.layer_num > self.trainable_layer:
            hidden_states = self.adapter(hidden_states)
        ###
        hidden_states = self.LayerNorm(hidden_states+input_tensor)
        return hidden_states

from torch.distributions.bernoulli import Bernoulli

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config, _) for _ in range(config.num_hidden_layers)])
        
    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, attention_show_flg=False):
                
        all_encoder_layers = []
        for layer_module in self.layer:
            if attention_show_flg == True:
                hidden_states, attention_probs = layer_module(hidden_states, attention_mask, attention_show_flg)
            elif attention_show_flg == False:
                hidden_states = layer_module(hidden_states, attention_mask, attention_show_flg)

            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        if attention_show_flg == True:
            return all_encoder_layers, attention_probs
        elif attention_show_flg == False:
            return all_encoder_layers

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertModel(PreTrainedBertModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        #self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, attention_show_flg=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attendion_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)

        if attention_show_flg == True:
            encoder_layer, attention_probs = self.encoder(embedding_output, extended_attention_mask, output_all_encoded_layers, attention_show_flg)

        elif attention_show_flg == False:
            encoder_layer = self.encoder(embedding_output, extended_attention_mask, output_all_encoded_layers, attention_show_flg)

        pooled_output = self.pooler(encoder_layer[-1])

        if not output_all_encoded_layers:
            encoder_layer = encoder_layer[-1]

        if attention_show_flg == True :
            return encoder_layer, pooled_output, attention_probs
        elif attention_show_flg == False :
            return encoder_layer, pooled_output

