#!/usr/bin/env python3

import argparse
from time import time
import librosa

from whisper import load_whisper, asr_from_array
from classification import load_bert, classify, id2label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', default='italian', help="select language, default='french'")
    parser.add_argument('-s', '--size', default='base', help="whisper size, base or small recommended")
    parser.add_argument('-b', '--blocksize', default=16000,
                        help="size of recording block, by default 16000, so 1 block "
                             "is one second", type=int)
    parser.add_argument('-f', '--filename', default=True, help="file to do the classification on, example files are in"
                                                               "'audio' directory")
    args = parser.parse_args()

    language = args.language
    filename = args.filename

    sr = 16000
    blocksize = args.blocksize  # 1s block = 16000

    processor, asr_model, forced_decoder_ids = load_whisper(args.language, args.size)

    # Loading classification model
    tokenizer, cls_model = load_bert()

    # extracting array from audio file
    array, _ = librosa.load(filename, sr=16000)

    # inferring the pipeline
    t0 = time()
    text = asr_from_array(array, processor, asr_model, forced_decoder_ids)
    print(text)

    res = classify(text, tokenizer, cls_model)
    print('#'*80)
    print("model size: ", args.size)
    print("infer time: ", time() - t0, )
    print(f"Classification: , {id2label[res]} ({res})")
    print('#'*80)

