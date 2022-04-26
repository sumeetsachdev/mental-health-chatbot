from pytorch_pretrained_bert import OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer
from itertools import chain

model = OpenAIGPTDoubleHeadsModel.from_pretrained('openai-gpt')
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

SPECIAL_TOKENS = ['<bos>', '<eos', '<speaker1>', '<speaker2>', '<pad>']

tokenizer.set_special_tokens(SPECIAL_TOKENS)
model.set_num_special_tokens(len(SPECIAL_TOKENS))

persona = [['i', 'like', 'playing', 'football', '.'],
           ['i', 'am', 'from', 'NYC', '.']]

history = [['hello', 'how', 'are', 'you', '?'],
           ['i', 'am', 'fine', 'thanks', '.']]

reply = ['great', 'to', 'hear']

bos, eos, speaker1, speaker2 = '<bos>', '<eos>', '<speaker1>', '<speaker2>'


def build_inputs(persona, history, reply):
    sequence = [[bos] + list(chain(*persona))] + history + [reply + [eos]]
    sequence = [sequence[0] +
                [ [speaker2 if (len(sequence) - i) % 2 else speaker1] + s for i,s in enumerate(sequence[1:])]]


    words = list(chain(*sequence))
    segments = [speaker2 if i % 2 else speaker1 for i,s in enumerate(sequence) for _ in s]
    position = list(range(len(words)))
    return words, segments, position, sequence

words, segments, position, sequence = build_inputs(persona, history, reply)

print(sequence)
