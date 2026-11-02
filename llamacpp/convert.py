# llama.cpp's convert_hf_to_gguf.py for Smaug-Decisions (and other Gemma 4 checkpoints with tied embeddings). Upstream's Gemma 4 text
# and mmproj converters do not skip an explicit `lm_head.weight`: the Smaug-Decisions checkpoint stores one although
# tie_word_embeddings=true (it is bit-identical to embed_tokens). The mmproj conversion then fails with
# "Can not map tensor 'lm_head.weight'" and the text GGUF gets a duplicate output.weight (+1.3 GB at bf16). This wrapper drops
# the tensor when the config says the embeddings are tied, then runs the stock converter unchanged (a no-op for checkpoints
# without it).
# usage: convert.py <llama.cpp dir> <model dir> <convert_hf_to_gguf.py args...>
#   e.g. convert.py llama.cpp Smaug-Decisions --outtype bf16 --outfile smaug-decisions-bf16.gguf
#        convert.py llama.cpp Smaug-Decisions --mmproj --outtype f16 --outfile mmproj-smaug-decisions-f16.gguf
import json
import os
import runpy
import sys


if len(sys.argv) < 3 or sys.argv[1].startswith('-'):
    sys.exit('usage: convert.py <llama.cpp dir> <model dir> <convert_hf_to_gguf.py args...>')
llama_dir = os.path.abspath(sys.argv.pop(1))
sys.path.insert(0, llama_dir)
sys.path.insert(0, os.path.join(llama_dir, 'gguf-py'))
from conversion import gemma  # noqa: E402


model_dir = next(a for a in sys.argv[1:] if not a.startswith('-') and os.path.isdir(a))
cfg = json.load(open(os.path.join(model_dir, 'config.json')))
tied = cfg.get('tie_word_embeddings', cfg.get('text_config', {}).get('tie_word_embeddings', False))
assert tied, 'lm_head is not tied to the embeddings: keep it (run the stock converter)'

for cls in (gemma.Gemma4Model, gemma.Gemma4VisionAudioModel):
    orig = cls.filter_tensors.__func__

    def filter_tensors(klass, item, orig=orig):
        return None if item[0] == 'lm_head.weight' else orig(klass, item)

    cls.filter_tensors = classmethod(filter_tensors)

sys.argv[0] = os.path.join(llama_dir, 'convert_hf_to_gguf.py')
runpy.run_path(sys.argv[0], run_name='__main__')
