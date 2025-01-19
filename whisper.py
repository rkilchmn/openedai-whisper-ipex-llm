#!/usr/bin/env python3
import os
import sys
import argparse

from transformers import pipeline
from typing import Optional, List
from fastapi import UploadFile, Form
from fastapi.responses import PlainTextResponse, JSONResponse
import uvicorn

# for ipex-llm implementation
from transformers import pipeline
from ipex_llm.transformers import AutoModelForSpeechSeq2Seq
from ipex_llm import optimize_model
from transformers.models.whisper import WhisperFeatureExtractor, WhisperTokenizer
from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig
from pathlib import Path

import openedai

QUANTIZATION_8BIT = '8-bit'
QUANTIZATION_4BIT = '4-bit'

pipe = None
app = openedai.OpenAIStub()

async def whisper(file, response_format: str, **kwargs):
    global pipe

    result = pipe(await file.read(), **kwargs)

    filename_noext, ext = os.path.splitext(file.filename)

    if response_format == "text":
        return PlainTextResponse(result["text"].strip(), headers={"Content-Disposition": f"attachment; filename={filename_noext}.txt"})

    elif response_format == "json":
        return JSONResponse(content={ 'text': result['text'].strip() }, media_type="application/json", headers={"Content-Disposition": f"attachment; filename={filename_noext}.json"})
    
    elif response_format == "verbose_json":
        chunks = result["chunks"]

        response = {
            "task": kwargs['generate_kwargs']['task'],
            #"language": "english",
            "duration": chunks[-1]['timestamp'][1],
            "text": result["text"].strip(),
        }
        if kwargs['return_timestamps'] == 'word':
            response['words'] = [{'word': chunk['text'].strip(), 'start': chunk['timestamp'][0], 'end': chunk['timestamp'][1] } for chunk in chunks ]
        else:
            response['segments'] = [{
                    "id": i,
                    #"seek": 0,
                    'start': chunk['timestamp'][0],
                    'end': chunk['timestamp'][1],
                    'text': chunk['text'].strip(),
                    #"tokens": [ ],
                    #"temperature": 0.0,
                    #"avg_logprob": -0.2860786020755768,
                    #"compression_ratio": 1.2363636493682861,
                    #"no_speech_prob": 0.00985979475080967
            } for i, chunk in enumerate(chunks) ]
        
        return JSONResponse(content=response, media_type="application/json", headers={"Content-Disposition": f"attachment; filename={filename_noext}_verbose.json"})

    elif response_format == "srt":
            def srt_time(t):
                return "{:02d}:{:02d}:{:06.3f}".format(int(t//3600), int(t//60)%60, t%60).replace(".", ",")

            return PlainTextResponse("\n".join([ f"{i}\n{srt_time(chunk['timestamp'][0])} --> {srt_time(chunk['timestamp'][1])}\n{chunk['text'].strip()}\n"
                for i, chunk in enumerate(result["chunks"], 1) ]), media_type="text/srt; charset=utf-8", headers={"Content-Disposition": f"attachment; filename={filename_noext}.srt"})

    elif response_format == "vtt":
            def vtt_time(t):
                return "{:02d}:{:06.3f}".format(int(t//60), t%60)
            
            return PlainTextResponse("\n".join(["WEBVTT\n"] + [ f"{vtt_time(chunk['timestamp'][0])} --> {vtt_time(chunk['timestamp'][1])}\n{chunk['text'].strip()}\n"
                for chunk in result["chunks"] ]), media_type="text/vtt; charset=utf-8", headers={"Content-Disposition": f"attachment; filename={filename_noext}.vtt"})


@app.post("/v1/audio/transcriptions")
@app.post("/inference/audio/transcriptions") # alternative path used by whisper writer
async def transcriptions(
        file: UploadFile,
        model: str = Form(...),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: Optional[str] = Form("json"),
        temperature: Optional[float] = Form(None),
        timestamp_granularities: List[str] = Form(["segment"])
    ):
    global pipe

    kwargs = {'generate_kwargs': {}}

    if language:
        kwargs['generate_kwargs']["language"] = language
        kwargs['generate_kwargs']["task"] = 'transcribe' # hf do not like having task on non multi language model
# May work soon, https://github.com/huggingface/transformers/issues/27317
#    if prompt:
#        kwargs["initial_prompt"] = prompt
    if temperature:
        kwargs['generate_kwargs']["temperature"] = temperature
        kwargs['generate_kwargs']['do_sample'] = True

    if response_format == "verbose_json" and 'word' in timestamp_granularities:
        kwargs['return_timestamps'] = 'word'
    else:
        kwargs['return_timestamps'] = response_format in ["verbose_json", "srt", "vtt"]

    return await whisper(file, response_format, **kwargs)


@app.post("/v1/audio/translations")
async def translations(
        file: UploadFile,
        model: str = Form(...),
        prompt: Optional[str] = Form(None),
        response_format: Optional[str] = Form("json"),
        temperature: Optional[float] = Form(None),
    ):
    global pipe

    kwargs = {'generate_kwargs': {"task": "translate"}}

# May work soon, https://github.com/huggingface/transformers/issues/27317
#    if prompt:
#        kwargs["initial_prompt"] = prompt
    if temperature:
        kwargs['generate_kwargs']["temperature"] = temperature
        kwargs['generate_kwargs']['do_sample'] = True

    kwargs['return_timestamps'] = response_format in ["verbose_json", "srt", "vtt"]

    return await whisper(file, response_format, **kwargs)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog='whisper.py',
        description='OpenedAI Whisper API Server',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-m', '--model', action='store', default="distil-whisper/distil-small.en", help="The model to use for transcription. Ex. distil-whisper/distil-medium.en")
    parser.add_argument('-d', '--device', action='store', default="xpu", help="Set the device for the model. Ex. 'xpu' for GPU or 'cpu' (default: 'xpu')")
    parser.add_argument('-t', '--dtype', action='store', default="auto", help="Set the torch data type for processing (float32, float16, bfloat16)")
    parser.add_argument('-q', '--quantization', action='store', default="", help=f"Enable model qunatization Ex. {QUANTIZATION_4BIT} or  {QUANTIZATION_8BIT} (default is off)")
    parser.add_argument('-P', '--port', action='store', default=8000, type=int, help="Server tcp port")
    parser.add_argument('-H', '--host', action='store', default='localhost', help="Host to listen on, Ex. 0.0.0.0")
    parser.add_argument('--preload', action='store_true', help="Preload model and exit.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    # enable qantization
    is_8bit = False
    is_4bit = False
    if args.quantization == QUANTIZATION_4BIT:
        is_4bit = True
    elif args.quantization == QUANTIZATION_8BIT:
        is_8bit = True

    # use ipex-llm model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, load_in_8bit=is_8bit, load_in_4bit=is_4bit)
    model.config.forced_decoder_ids = None

    # With only one line to enable IPEX-LLM optimize on a pytorch model
    model = optimize_model(model)
    model = model.to(args.device)

    # Configure pipeline
    pipe = pipeline(
      "automatic-speech-recognition",
      model=model,
      feature_extractor= WhisperFeatureExtractor.from_pretrained(args.model),
      tokenizer= WhisperTokenizer.from_pretrained(args.model),
      chunk_length_s=30,
      device=args.device
    )

    if args.preload:
        sys.exit(0)

    app.register_model('whisper-1', args.model)

    uvicorn.run(app, host=args.host, port=args.port) # , root_path=cwd, access_log=False, log_level="info", ssl_keyfile="cert.pem", ssl_certfile="cert.pem")
