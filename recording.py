import queue
import sys
from time import sleep

import sounddevice as sd
import soundfile as sf
import numpy as np


q = queue.Queue()


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


def background(sr, device, blocksize):
    record = np.array([])

    with sd.InputStream(samplerate=sr, device=device,
                        channels=1, callback=callback, blocksize=blocksize, ):
        print('#' * 80)
        print('recording background 5 sec')
        print('#' * 80)
        while record.shape[0] <= sr * 5:
            processed = q.get()
            std = np.std(processed)
            print(std)
            if not len(record) > 0:
                record = np.array(processed)
            record = np.vstack((record, processed))
        print('#' * 80)


def autostop(sr, device, blocksize, stopcrit=0.01):

    std = 1
    record = np.array([])
    with sd.InputStream(samplerate=sr, device=device,
                        channels=1, callback=callback, blocksize=blocksize, ):
        print('#' * 80)
        # print('record start in 1 sec, Ctrl+C to stop manually, ')
        # print('#' * 80)
        # sleep(1)
        print("starting")
        print('#' * 80)
        # sleep(0.5)
        while std >= stopcrit and sr * 9 >= len(record):
            processed = q.get()
            std = np.std(processed)
            print(std)
            if not len(record) > 0:
                record = np.array(processed)
            record = np.vstack((record, processed))
        print("ending")
        print('#' * 80)

    return record.T[0]


def autostop_save(sr, device, blocksize, filename, stopcrit=0.01):
    std = 1
    record = np.array([])

    with sf.SoundFile(filename, mode='w', samplerate=sr,
                      channels=1) as file:
        print('#' * 80)
        print("starting record")
        print('#' * 80)
        with sd.InputStream(samplerate=sr, device=device,
                            channels=1, callback=callback, blocksize=blocksize):

            while std >= stopcrit and record.shape[0] <= sr * 9:
                processed = q.get()
                file.write(processed)
                std = np.std(processed)
                print(std)
                if not len(record) > 0:
                    record = np.array(processed)
                record = np.vstack((record, processed))
    print('#' * 80)
    print("ending")
    print('#' * 80)

    return record.T[0]


if __name__ == "__main__":
    print(sd.query_devices())
    device = int(input("device: "))
    autostop_save(16000, device, 16000, "test.wav")
