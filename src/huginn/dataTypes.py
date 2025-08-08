from typing import Literal, List, TypeAlias
from dataclasses import dataclass

FastTokenisers = Literal['AlbertTokenizer', 'BartTokenizer', 'BarthezTokenizer', 'BertTokenizer', 'BigBirdTokenizer', 'BlenderbotTokenizer', 'CamembertTokenizer', 'CLIPTokenizer', 'CodeGenTokenizer', 'ConvBertTokenizer', 'DebertaTokenizer', 'DebertaV2Tokenizer', 'DistilBertTokenizer', 'DPRReaderTokenizer', 'DPRQuestionEncoderTokenizer', 'DPRContextEncoderTokenizer', 'ElectraTokenizer', 'FNetTokenizer', 'FunnelTokenizer', 'GPT2Tokenizer', 'HerbertTokenizer', 'LayoutLMTokenizer', 'LayoutLMv2Tokenizer', 'LayoutLMv3Tokenizer', 'LayoutXLMTokenizer', 'LongformerTokenizer', 'LEDTokenizer', 'LxmertTokenizer', 'MarkupLMTokenizer', 'MBartTokenizer', 'MBart50Tokenizer', 'MPNetTokenizer', 'MobileBertTokenizer', 'MvpTokenizer', 'NllbTokenizer', 'OpenAIGPTTokenizer', 'PegasusTokenizer', 'Qwen2Tokenizer', 'RealmTokenizer', 'ReformerTokenizer', 'RemBertTokenizer', 'RetriBertTokenizer', 'RobertaTokenizer', 'RoFormerTokenizer', 'SeamlessM4TTokenizer', 'SqueezeBertTokenizer', 'T5Tokenizer', 'UdopTokenizer', 'WhisperTokenizer', 'XLMRobertaTokenizer', 'XLNetTokenizer', 'SplinterTokenizer', 'XGLMTokenizer', 'LlamaTokenizer', 'CodeLlamaTokenizer', 'GemmaTokenizer', 'Phi3Tokenizer']

SlowTokenisers = Literal["aimv2", "albert", "align", "arcee", "aria", "aya_vision", "bark", "bart", "barthez", "bartpho", "bert", "bert-generation", "bert-japanese", "bertweet", "big_bird", "bigbird_pegasus", "biogpt", "bitnet", "blenderbot", "blenderbot-small", "blip", "blip-2", "bloom", "bridgetower", "bros", "byt5", "camembert", "canine", "chameleon", "chinese_clip", "clap", "clip", "clipseg", "clvp", "code_llama", "codegen", "cohere", "cohere2", "colpali", "colqwen2", "convbert", "cpm", "cpmant", "ctrl", "data2vec-audio", "data2vec-text", "dbrx", "deberta", "deberta-v2", "deepseek_v2", "deepseek_v3", "deepseek_vl", "deepseek_vl_hybrid", "dia", "diffllama", "distilbert", "dpr", "electra", "emu3", "ernie", "ernie4_5", "ernie4_5_moe", "ernie_m", "esm", "exaone4", "falcon", "falcon_mamba", "fastspeech2_conformer", "flaubert", "fnet", "fsmt", "funnel", "gemma", "gemma2", "gemma3", "gemma3_text", "gemma3n", "gemma3n_text", "git", "glm", "glm4", "glm4_moe", "glm4v", "gpt-sw3", "gpt2", "gpt_bigcode", "gpt_neo", "gpt_neox", "gpt_neox_japanese", "gpt_oss", "gptj", "gptsan-japanese", "granite", "granitemoe", "granitemoehybrid", "granitemoeshared", "grounding-dino", "groupvit", "helium", "herbert", "hubert", "ibert", "idefics", "idefics2", "idefics3", "instructblip", "instructblipvideo", "internvl", "jamba", "janus", "jetmoe", "jukebox", "kosmos-2", "layoutlm", "layoutlmv2", "layoutlmv3", "layoutxlm", "led", "lilt", "llama", "llama4", "llama4_text", "llava", "llava_next", "llava_next_video", "llava_onevision", "longformer", "longt5", "luke", "lxmert", "m2m_100", "mamba", "mamba2", "marian", "mbart", "mbart50", "mega", "megatron-bert", "mgp-str", "minimax", "mistral", "mixtral", "mllama", "mluke", "mm-grounding-dino", "mobilebert", "modernbert", "moonshine", "moshi", "mpnet", "mpt", "mra", "mt5", "musicgen", "musicgen_melody", "mvp", "myt5", "nemotron", "nezha", "nllb", "nllb-moe", "nystromformer", "olmo", "olmo2", "olmoe", "omdet-turbo", "oneformer", "openai-gpt", "opt", "owlv2", "owlvit", "paligemma", "pegasus", "pegasus_x", "perceiver", "persimmon", "phi", "phi3", "phimoe", "phobert", "pix2struct", "pixtral", "plbart", "prophetnet", "qdqbert", "qwen2", "qwen2_5_omni", "qwen2_5_vl", "qwen2_audio", "qwen2_moe", "qwen2_vl", "qwen3", "qwen3_moe", "rag", "realm", "recurrent_gemma", "reformer", "rembert", "retribert", "roberta", "roberta-prelayernorm", "roc_bert", "roformer", "rwkv", "seamless_m4t", "seamless_m4t_v2", "shieldgemma2", "siglip", "siglip2", "smollm3", "speech_to_text", "speech_to_text_2", "speecht5", "splinter", "squeezebert", "stablelm", "starcoder2", "switch_transformers", "t5", "t5gemma", "tapas", "tapex", "transfo-xl", "tvp", "udop", "umt5", "video_llava", "vilt", "vipllava", "visual_bert", "vits", "voxtral", "wav2vec2", "wav2vec2-bert", "wav2vec2-conformer", "wav2vec2_phoneme", "whisper", "xclip", "xglm", "xlm", "xlm-prophetnet", "xlm-roberta", "xlm-roberta-xl", "xlnet", "xlstm", "xmod", "yoso", "zamba", "zamba2"]

ModelTypes = Literal[
    "decision", 
    "text", 
    "speech", 
    "translation", 
    "visual"
    ]

Device = Literal["cpu", "cuda"]
@dataclass
class TokenizerType:
    fast: List[FastTokenisers]
    slow: List[SlowTokenisers]

@dataclass
class ModelSpec:
    name: str
    type: ModelTypes
    context: int
    version: str
    description: str
    id: str
    memory: int
    size: int
    tokenizers: TokenizerType

Models: List[ModelSpec] = [
    ModelSpec(
        name="Command R",
        type="text",
        context=70,
        version="08-2024",
        description="A command R model from Cohere Labs.",
        id="CohereLabs/c4ai-command-r-plus-08-2024",
        memory=13_000_000_000,
        size=20_000_000_000,
        tokenizers=TokenizerType(
            fast=["GPT2Tokenizer", "Qwen2Tokenizer", "Phi3Tokenizer"],
            slow=["gpt2", "qwen2", "phi3"],
        ),
    ),
    ModelSpec(
        name="Llama 3",
        type="text",
        context=8,
        version="3.1",
        description="A Llama model from Meta.",
        id="meta-llama/Llama-3.1-8B",
        memory=2_000_000_000,
        size=5_000_000_000,
        tokenizers=TokenizerType(
            fast=["LlamaTokenizer", "CodeLlamaTokenizer"],
            slow=["llama", "code_llama"],
        ),
    ),
    ModelSpec(
        name="Gemma 2B",
        type="text",
        context=8,
        version="1.1",
        description="A Gemma model from Google.",
        id="google/gemma-2b",
        memory=1_500_000_000,
        size=3_000_000_000,
        tokenizers=TokenizerType(
            fast=["GemmaTokenizer"],
            slow=["gemma"],
        ),
    ),
    ModelSpec(
        name="Phi 3",
        type="text",
        context=8,
        version="3.0",
        description="A Phi-3 model from Microsoft.",
        id="microsoft/phi-3-mini-4k-instruct",
        memory=1_000_000_000,
        size=2_000_000_000,
        tokenizers=TokenizerType(
            fast=["Phi3Tokenizer"],
            slow=["phi3"],
        ),
    ),
    ModelSpec(
        name="GPT 2",
        type="text",
        context=8,
        version="0.0",
        description="A very small GPT-2 model for testing.",
        id="sshleifer/tiny-gpt2",
        memory=50_000_000,
        size=50_000_000,
        tokenizers=TokenizerType(
            fast=["GPT2Tokenizer"],
            slow=["gpt2"],
        ),
    ),
]
modelIDs: List[str] = [model.id for model in Models]
ModelID: TypeAlias = Literal[*modelIDs] # type: ignore