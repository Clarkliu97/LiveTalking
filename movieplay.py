from io import BytesIO
import cv2
from av import AudioFrame, VideoFrame
import av
import asyncio
import threading
import time
import soundfile as sf
import numpy as np
import resampy


class AudioProcess:
    def __init__(self, file_or_url: str, opt):
        self.fps = opt.fps
        self.sample_rate = 16_000
        self.chunk = self.sample_rate // self.fps
        self.input_stream = open(file_or_url, 'rb')
        
        stream = self.__create_bytes_stream(self.input_stream)
        streamlen = stream.shape[0]
        idx=0
        self.frames = []
        while streamlen >= self.chunk:
            self.frames.append(stream[idx:idx+self.chunk])
            streamlen -= self.chunk
            idx += self.chunk

    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        print(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            print(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            print(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream


class MoviePlay:
    def __init__(self, opt):
        self.movefile = 'test-v.mp4'
        self.audiofile = 'test-a.mp3'
        self.opt = opt
        self.cap = cv2.VideoCapture(self.movefile)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # self.audio_frames = read_mp3_frames(self.audiofile, frame_duration_ms=1000/self.fps)
        # self.audio_frame_idx = 0
        self.audio = AudioProcess(self.audiofile, opt)
        self.audio_frames = self.audio.frames
        print("total_frames:", len(self.audio_frames))
        self.played_audio = False
        

    def play(self, quit_event, loop=None, audio_track=None, video_track=None):
        while not quit_event.is_set():
            ret ,frame = self.cap.read()
            if ret == False:
                if loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            new_frame = VideoFrame.from_ndarray(frame, format='bgr24')
            asyncio.run_coroutine_threadsafe(video_track._queue.put(new_frame), loop)

            if audio_track and not self.played_audio:
                # frame = self.audio_frames[self.audio_frame_idx % len(self.audio_frames)]
                for audio_frame in self.audio_frames:
                    # buffer = frame.planes[0]
                    # data = np.frombuffer(buffer, dtype=np.int16)
                    # data = data / np.max(np.abs(data))
                    # print(data.shape)
                    # audio_frame = AudioFrame(samples=data.shape[0], layout="mono", format='s16')
                    # new_frame.planes[0].update(data.tobytes())
                    # new_frame.sample_rate=16000
                    # asyncio.run_coroutine_threadsafe(audio_track._queue.put(audio_frame), loop)

                    frame = audio_frame
                    frame = (frame * 32767).astype(np.int16)
                    new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                    new_frame.planes[0].update(frame.tobytes())
                    new_frame.sample_rate=16000
                    # if audio_track._queue.qsize()>10:
                    #     time.sleep(0.1)
                    asyncio.run_coroutine_threadsafe(audio_track._queue.put(new_frame), loop)

                self.played_audio = True


    def render(self, quit_event, loop=None, audio_track=None, video_track=None):
        # self.render
        play_thread = threading.Thread(target=self.play, args=(quit_event, loop, audio_track, video_track))
        play_thread.start()

        print("movieplay.py: render() end")
