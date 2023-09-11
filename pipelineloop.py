#!/usr/bin/env python

import argparse
from sounddevice import query_devices

from whisper import load_whisper, asr_from_array
from recording import autostop, autostop_save, background
from classification import load_bert, classify, id2label


def language(lg):
    if lg.lower() in "french":
        return "french"
    if lg.lower() in "italian":
        return "italian"
    else:
        raise ValueError("language must be 'french'/'fr' or 'italian'/'it'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', default='italian', help="select language, default='french'")
    parser.add_argument('-s', '--size', default='base', help="whisper size, base or small recommended")
    # parser.add_argument('-d', '--device', default=0, help="device to use for microphone, list of devices type 'python -m sounddevice'", type=int)
    parser.add_argument('-b', '--blocksize', default=16000, help="size of recording block, by default 16000, so 1 block is one second", type=int)
    parser.add_argument('-f', '--filename', default=True, help="If wishing to save audio in a file, ex 'test.wav'")
    args = parser.parse_args()
    language = args.language

    sr = 16000
    blocksize = args.blocksize # 1s block = 16000

    # select input device for audio
    print(query_devices())
    device = int(input("device: "))

    background(sr, device, blocksize)
    stopcrit = float(input("Enter stop criteria relative to background noise: "))

    # Loading whisper
    processor, asr_model, forced_decoder_ids = load_whisper(args.language, args.size)

    # Loading classification
    tokenizer, cls_model = load_bert()

    while True:
        # asr using microphone, max 10 seconds
        key = input("press enter to speak, s to stop: ")
        if key == "s":
            break
        if args.filename:
            record = autostop(sr, device, args.blocksize, stopcrit=stopcrit)
        else:
            record = autostop_save(sr, device, args.blocksize, args.filename)

        # print(record)

        text = asr_from_array(record, processor, asr_model, forced_decoder_ids)
        print(text, flush=True)

        res = classify(text, tokenizer, cls_model)
        print(f"Classification: , {id2label[res]} ({res})")

