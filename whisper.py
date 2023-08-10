from transformers import WhisperProcessor, WhisperForConditionalGeneration


def load_whisper(language='french', size='base'):
    """
    load model and processor.
    :param language: language for asr, italian or french to english
    :return: processor, whisper model, forced_decoder_ids
    """
    processor = WhisperProcessor.from_pretrained("openai/whisper-"+size)
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-"+size)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="translate")
    return processor, model, forced_decoder_ids


def asr_from_file(file_name, processor, model, forced_decoder_ids):
    """
    return transcription from audio-file in French in english
    :param file_name: name of the file (any format in theory)
    :return: Audio transcription of the file.
    """
    # load mp3 file
    array, sampling_rate = librosa.load(file_name, sr=16000)
    print(array.shape)
    input_features = processor(array, sampling_rate=sampling_rate, return_tensors="pt").input_features
    # generate token ids
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    # print(transcription)
    return transcription[0]


def asr_from_array(array, processor, model, forced_decoder_ids):
    input_features = processor(array, sampling_rate=16000, return_tensors="pt").input_features
    # generate token ids
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    # print(transcription)
    return transcription[0]


