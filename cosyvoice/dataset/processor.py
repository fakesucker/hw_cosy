# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
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
import logging
import random

import pyarrow.parquet as pq
from io import BytesIO
import json
import numpy as np
import whisper
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pyworld as pw
from cosyvoice.utils.onnx import embedding_extractor, online_feature

AUDIO_FORMAT_SETS = {'flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'}

def jsonl_opener(data, mode='train', tts_data={}, is_use_wav_path=False):
    """ Read jsonl files and yield samples for LM training only
        Simplified version without audio_data and embeddings
        
        Args:
            data(Iterable[str]): url or local file list (jsonl files)
            mode: 'train' or other modes
            tts_data: dict for tts mode
            
        Returns:
            Iterable[{utt, text, speech_token, ...}]
    """
    for sample in data:
        assert 'src' in sample
        url = sample['src']
        try:
            with open(url, 'r', encoding='utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json_data = json.loads(line)
                        # 基于上游信息创建当前样本副本
                        current_sample = dict(sample)
                        # Map jsonl fields to expected fields
                        current_sample.update({
                            'utt': json_data.get('key', json_data.get('utt', '')),
                            'text': json_data.get('txt', json_data.get('text', ''))
                        })
                        
                        # Handle speech_token (code field)
                        speech_token = json_data.get('code', json_data.get('speech_token', []))
                        # Convert to list if it's not already
                        if isinstance(speech_token, str):
                            import ast
                            speech_token = ast.literal_eval(speech_token)
                        current_sample['speech_token'] = speech_token
                        
                        # Handle duration if available (for filtering)
                        if 'duration' in json_data:
                            current_sample['duration'] = json_data['duration']
                        else:
                            current_sample.pop('duration', None)

                        if 'prompt_text' in json_data:
                            current_sample['prompt_text'] = json_data.get('prompt_text', '')
                        else:
                            current_sample.pop('prompt_text', None)
                        if 'prompt_wav' in json_data:
                            current_sample['prompt_wav'] = json_data.get('prompt_wav', '')
                        else:
                            current_sample.pop('prompt_wav', None)
                        if 'prompt_code' in json_data or 'prompt_speech_token' in json_data:
                            prompt_token = json_data.get('prompt_code', json_data.get('prompt_speech_token', []))
                            if isinstance(prompt_token, str):
                                import ast
                                prompt_token = ast.literal_eval(prompt_token)
                            current_sample['prompt_speech_token'] = prompt_token
                        else:
                            current_sample.pop('prompt_speech_token', None)
                            
                        if is_use_wav_path and 'wav_path' in json_data:
                            wav_path = json_data['wav_path']
                            try:
                                waveform, sample_rate = torchaudio.load(wav_path)
                                
                                # 转换为单声道（如果是多声道）
                                if waveform.shape[0] > 1:
                                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                                
                                # 重采样到16000Hz（如果需要）
                                if sample_rate != 16000:
                                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                                    waveform = resampler(waveform)
                                
                                # 转换为1D tensor（去除channel维度）
                                waveform = waveform.squeeze(0)  # shape: (length,)
                                
                                # 归一化到[-1, 1]（如果还没有）
                                max_val = waveform.abs().max()
                                if max_val > 1.0:
                                    waveform = waveform / max_val
                                
                                wav_length = waveform.shape[0]
                                # 确保tensor不需要梯度，可以安全地在多进程间传递
                                if waveform.requires_grad:
                                    waveform = waveform.detach()
                                current_sample['speech'] = waveform
                                current_sample['wav_length'] = wav_length
                            except Exception as e:
                                logging.warning(f'Failed to load audio from {wav_path}: {e}')
                                continue
                        else:
                            current_sample.pop('speech', None)
                            current_sample.pop('wav_length', None)
                        
                        # Handle reject_speech_token for DPO
                        if 'reject_code' in json_data or 'reject_speech_token' in json_data:
                            reject_token = json_data.get('reject_code', json_data.get('reject_speech_token', []))
                            if isinstance(reject_token, str):
                                import ast
                                reject_token = ast.literal_eval(reject_token)
                            current_sample['reject_speech_token'] = reject_token
                        else:
                            current_sample.pop('reject_speech_token', None)
                        
                        if mode == 'train':
                            yield current_sample
                        else:
                            for index, text in enumerate(tts_data.get(current_sample['utt'], [])):
                                infer_sample = dict(current_sample)
                                infer_sample.update({'tts_index': index, 'tts_text': text})
                                yield infer_sample
                    except json.JSONDecodeError as e:
                        logging.warning('Failed to parse JSON line in {}, ex info {}'.format(url, e))
                        continue
        except Exception as ex:
            logging.warning('Failed to open {}, ex info {}'.format(url, ex))



def parquet_opener(data, mode='train'):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
    for sample in data:
        assert 'src' in sample
        url = sample['src']
        try:
            for df in pq.ParquetFile(url).iter_batches(batch_size=64):
                df = df.to_pandas()
                for i in range(len(df)):
                    sample.update(dict(df.loc[i]))
                    # NOTE do not return sample directly, must initialize a new dict
                    yield {**sample}
        except Exception as ex:
            logging.warning('Failed to open {}, ex info {}'.format(url, ex))


def filter(data,
           max_length=10240,
           min_length=10,
           token_max_length=200,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1,
           mode='train'):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            data: Iterable[{key, wav, label, sample_rate}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        # Calculate num_frames from speech_token length (25 tokens per second)
        # speech_token is at 25Hz, so num_frames = len(speech_token) / 25 * 100 (frames per 10ms)
        if 'speech_token' in sample and len(sample['speech_token']) > 0:
            num_frames = len(sample['speech_token']) / 25.0 * 100.0
        elif 'duration' in sample:
            num_frames = sample['duration'] * 100
        else:
            continue
        if num_frames < min_length:
            continue
        if num_frames > max_length:
            continue
        if len(sample['text_token']) < token_min_length:
            continue
        if len(sample['text_token']) > token_max_length:
            continue
        if online_feature is False and len(sample['speech_token']) == 0:
            continue
        if online_feature is False and 'reject_speech_token' in sample and len(sample['reject_speech_token']) == 0:
            continue
        if num_frames != 0:
            if len(sample['text_token']) / num_frames < min_output_input_ratio:
                continue
            if len(sample['text_token']) / num_frames > max_output_input_ratio:
                continue
        yield sample


def resample(data, resample_rate=22050, min_sample_rate=16000, mode='train'):
    """ Resample data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['speech']
        if sample_rate != resample_rate:
            if sample_rate < min_sample_rate:
                continue
            sample['sample_rate'] = resample_rate
            sample['speech'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        max_val = sample['speech'].abs().max()
        if max_val > 1:
            sample['speech'] /= max_val
        yield sample


def truncate(data, truncate_length=24576, mode='train'):
    """ Truncate data.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            truncate_length: truncate length

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        waveform = sample['speech']
        if waveform.shape[1] > truncate_length:
            start = random.randint(0, waveform.shape[1] - truncate_length)
            waveform = waveform[:, start: start + truncate_length]
        else:
            waveform = torch.concat([waveform, torch.zeros(1, truncate_length - waveform.shape[1])], dim=1)
        sample['speech'] = waveform
        yield sample


def compute_fbank(data,
                  feat_extractor,
                  num_frames=-1,
                  mode='train'):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        assert 'utt' in sample
        assert 'text_token' in sample
        # NOTE in cosyvoice2/3, we support online token extraction, so we need to align speech to 25hz first
        if num_frames != -1:
            index = int(np.ceil(sample['speech'].shape[1] / num_frames))
            sample['speech'] = torch.concat([sample['speech'], torch.zeros(1, index * num_frames - sample['speech'].shape[1])], dim=1)
        sample['speech_feat'] = feat_extractor(sample['speech']).squeeze(dim=0).transpose(0, 1)
        yield sample


def compute_whisper_fbank(data, num_frames=-1, mode='train'):
    """ Extract whisper fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        if num_frames != -1:
            assert sample['speech'].shape[1] % num_frames == 0, 'speech length is not aligned with speech_token'
        sample['speech_16k'] = torchaudio.transforms.Resample(orig_freq=sample['sample_rate'], new_freq=16000)(sample['speech'])
        sample['whisper_feat'] = whisper.log_mel_spectrogram(sample['speech_16k'], n_mels=128).squeeze(dim=0).transpose(0, 1)
        yield sample


def compute_f0(data, sample_rate, hop_size, mode='train'):
    """ Extract f0

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    frame_period = hop_size * 1000 / sample_rate
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        assert 'utt' in sample
        assert 'text_token' in sample
        waveform = sample['speech']
        _f0, t = pw.harvest(waveform.squeeze(dim=0).numpy().astype('double'), sample_rate, frame_period=frame_period)
        if sum(_f0 != 0) < 5:  # this happens when the algorithm fails
            _f0, t = pw.dio(waveform.squeeze(dim=0).numpy().astype('double'), sample_rate, frame_period=frame_period)  # if harvest fails, try dio
        f0 = pw.stonemask(waveform.squeeze(dim=0).numpy().astype('double'), _f0, t, sample_rate)
        f0 = F.interpolate(torch.from_numpy(f0).view(1, 1, -1), size=sample['speech_feat'].shape[0], mode='linear').view(-1)
        sample['pitch_feat'] = f0
        yield sample


def parse_embedding(data, normalize, mode='train'):
    """ Parse utt_embedding/spk_embedding

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        if 'utt_embedding' not in sample and 'spk_embedding' not in sample:
            sample['speech_16k'] = torchaudio.transforms.Resample(orig_freq=sample['sample_rate'], new_freq=16000)(sample['speech'])
            embedding = embedding_extractor.inference(sample['speech_16k'])
            sample['spk_embedding'] = sample['utt_embedding'] = embedding
        else:
            sample['utt_embedding'] = torch.tensor(sample['utt_embedding'], dtype=torch.float32)
            sample['spk_embedding'] = torch.tensor(sample['spk_embedding'], dtype=torch.float32)
        if normalize:
            sample['utt_embedding'] = F.normalize(sample['utt_embedding'], dim=0)
            sample['spk_embedding'] = F.normalize(sample['spk_embedding'], dim=0)
        yield sample


def tokenize(data, get_tokenizer, allowed_special, speaker_id='<|spk_1|>', mode='train', is_use_speaker_id=False):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            data: Iterable[{key, wav, txt, sample_rate}]

        Returns:
            Iterable[{key, wav, txt, tokens, label, sample_rate}]
    """
    tokenizer = get_tokenizer()
    for sample in data:
        assert 'text' in sample
        sample['text_token'] = tokenizer.encode(sample['text'], allowed_special=allowed_special)
        if is_use_speaker_id:
            speaker_id_token = tokenizer.encode(speaker_id, allowed_special=allowed_special)
            if speaker_id_token and sample['text_token'][:len(speaker_id_token)] != speaker_id_token:
                sample['text_token'] = speaker_id_token + sample['text_token']
        if 'prompt_text' in sample:
            sample['prompt_text_token'] = tokenizer.encode(sample['prompt_text'], allowed_special=allowed_special)
        if 'instruct' in sample:
            sample['instruct_token'] = tokenizer.encode(sample['instruct'], allowed_special=allowed_special)
        yield sample


def shuffle(data, shuffle_size=10000, mode='train'):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, feat, label}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, feat, label}]
    """
    buf = []
    yield_size = int(shuffle_size / 2)
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf[:yield_size]:
                yield x
            buf = buf[yield_size:]
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def sort(data, sort_size=500, mode='train'):
    """ Sort the data by speech_token length for LM training.
        Sort is used after shuffle and before batch, so we can group
        utts with similar lengths into a batch, and `sort_size` should
        be less than `shuffle_size`

        Args:
            data: Iterable[{utt, text_token, speech_token}]
            sort_size: buffer size for sort

        Returns:
            Iterable[{utt, text_token, speech_token}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            # Sort by speech_token length
            buf.sort(key=lambda x: len(x.get('speech_token', [])))
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=lambda x: len(x.get('speech_token', [])))
    for x in buf:
        yield x


def static_batch(data, batch_size=16):
    """ Static batch the data by `batch_size`

        Args:
            data: Iterable[{key, feat, label}]
            batch_size: batch size

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dynamic_batch(data, max_frames_in_batch=24000, mode='train'):
    """ Dynamic batch the data until the total frames in batch
        reach `max_frames_in_batch` (for LM training, using speech_token length)

        Args:
            data: Iterable[{utt, text_token, speech_token}]
            max_frames_in_batch: max_frames in one batch (based on speech_token length)

        Returns:
            Iterable[List[{utt, text_token, speech_token}]]
    """
    buf = []
    longest_token_len = 0
    for sample in data:
        assert 'speech_token' in sample
        new_token_len = len(sample['speech_token'])
        longest_token_len = max(longest_token_len, new_token_len)
        # Estimate frames: speech_token is at 25Hz, so frames = token_len / 25 * 100
        frames_after_padding = (longest_token_len / 25.0 * 100.0) * (len(buf) + 1)
        if frames_after_padding > max_frames_in_batch:
            yield buf
            buf = [sample]
            longest_token_len = new_token_len
        else:
            buf.append(sample)
    if len(buf) > 0:
        yield buf


def batch(data, batch_type='static', batch_size=16, max_frames_in_batch=12000, mode='train'):
    """ Wrapper for static/dynamic batch
    """
    if batch_type == 'static':
        return static_batch(data, batch_size)
    elif batch_type == 'dynamic':
        return dynamic_batch(data, max_frames_in_batch)
    else:
        logging.fatal('Unsupported batch type {}'.format(batch_type))


def padding(data, use_spk_embedding=False, mode='train', gan=False, dpo=False):
    """ Padding the data and spk_embedding into training data for LM training only
        Simplified version without speech_feat, pitch_feat, embeddings

        Args:
            data: Iterable[List[{utt, text, text_token, speech_token}]]
            use_spk_embedding: whether to use spk_embedding
        Returns:
            Iterable[Dict with batched data]
    """
    for sample in data:
        assert isinstance(sample, list)
        # Sort by speech_token length (descending) for efficient batching
        speech_token_lens = [len(x['speech_token']) for x in sample]
        order = torch.argsort(torch.tensor(speech_token_lens), descending=True)

        utts = [sample[i]['utt'] for i in order]
        text = [sample[i]['text'] for i in order]
        
        # Pad text_token
        text_token = [torch.tensor(sample[i]['text_token'], dtype=torch.long) for i in order]
        text_token_len = torch.tensor([len(sample[i]['text_token']) for i in order], dtype=torch.int32)
        text_token = pad_sequence(text_token, batch_first=True, padding_value=0)
        
        # Pad speech_token
        speech_token = [torch.tensor(sample[i]['speech_token'], dtype=torch.long) for i in order]
        speech_token_len = torch.tensor([len(sample[i]['speech_token']) for i in order], dtype=torch.int32)
        speech_token = pad_sequence(speech_token, batch_first=True, padding_value=0)
        
        # Create batch dict with only LM training needed fields
        batch = {
            "utts": utts,
            "text": text,
            "text_token": text_token,
            "text_token_len": text_token_len,
            "speech_token": speech_token,
            "speech_token_len": speech_token_len,
        }

        if all('prompt_text_token' in sample[i] and 'prompt_speech_token' in sample[i] for i in order):
            prompt_text = [sample[i].get('prompt_text', '') for i in order]
            prompt_wav = [sample[i].get('prompt_wav', '') for i in order]
            prompt_text_token = [torch.tensor(sample[i]['prompt_text_token'], dtype=torch.long) for i in order]
            prompt_text_token_len = torch.tensor([len(sample[i]['prompt_text_token']) for i in order], dtype=torch.int32)
            prompt_text_token = pad_sequence(prompt_text_token, batch_first=True, padding_value=0)
            prompt_speech_token = [torch.tensor(sample[i]['prompt_speech_token'], dtype=torch.long) for i in order]
            prompt_speech_token_len = torch.tensor([len(sample[i]['prompt_speech_token']) for i in order], dtype=torch.int32)
            prompt_speech_token = pad_sequence(prompt_speech_token, batch_first=True, padding_value=0)
            batch.update({
                "prompt_text": prompt_text,
                "prompt_wav": prompt_wav,
                "prompt_text_token": prompt_text_token,
                "prompt_text_token_len": prompt_text_token_len,
                "prompt_speech_token": prompt_speech_token,
                "prompt_speech_token_len": prompt_speech_token_len,
            })
        
        # 批量提取 spk_embedding
        if use_spk_embedding:
            # import pdb; pdb.set_trace()
            # 检查是否有 speech 和 wav_length 字段
            batch_samples = [sample[i] for i in order]
            spk_embedding = campplus_embedding(batch_samples, mode=mode)
            # spk_embedding shape: (batch_size, 192)，已经是batch格式，不需要padding
            batch['spk_embedding'] = spk_embedding

        # Handle DPO reject_speech_token if needed
        if dpo is True:
            reject_speech_token = [torch.tensor(sample[i]['reject_speech_token'], dtype=torch.long) for i in order]
            reject_speech_token_len = torch.tensor([len(sample[i]['reject_speech_token']) for i in order], dtype=torch.int32)
            reject_speech_token = pad_sequence(reject_speech_token, batch_first=True, padding_value=0)
            batch['reject_speech_token'] = reject_speech_token
            batch['reject_speech_token_len'] = reject_speech_token_len
        
        yield batch
