# server.py
from flask import Flask, render_template,send_from_directory,request, jsonify
from flask_sockets import Sockets
import base64
import time
import json
#import gevent
#from gevent import pywsgi
#from geventwebsocket.handler import WebSocketHandler
import os
import re
import numpy as np
from threading import Thread,Event
#import multiprocessing
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.rtcrtpsender import RTCRtpSender
from webrtc import HumanPlayer

import argparse
import random

import shutil
import asyncio
import torch

import uuid

app = Flask(__name__)
#sockets = Sockets(app)
nerfreals = {}
opt = None
model = None
avatar = None

#####webrtc###############################
pcs = set()

def build_nerfreal(sessionid):
    opt.sessionid=sessionid
    if opt.model == 'wav2lip':
        from lipreal import LipReal
        nerfreal = LipReal(opt,model,avatar)
    elif opt.model == 'musetalk':
        from musereal import MuseReal
        nerfreal = MuseReal(opt,model,avatar)
    elif opt.model == 'ernerf':
        from nerfreal import NeRFReal
        nerfreal = NeRFReal(opt,model,avatar)
    return nerfreal

#@app.route('/offer', methods=['POST'])
async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    if len(nerfreals) >= opt.max_session:
        print('reach max session')
        return -1
    sessionid = uuid.uuid1().int
    print('sessionid=',sessionid)
    nerfreals[sessionid] = None
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid)
    nerfreals[sessionid] = nerfreal
    
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            del nerfreals[sessionid]
        if pc.connectionState == "closed":
            pcs.discard(pc)
            del nerfreals[sessionid]

    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)
    capabilities = RTCRtpSender.getCapabilities("video")
    preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
    transceiver = pc.getTransceivers()[1]
    transceiver.setCodecPreferences(preferences)

    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    #return jsonify({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid":sessionid}
        ),
    )

async def human(request):
    params = await request.json()

    sessionid = params.get('sessionid',0)
    if params.get('interrupt'):
        nerfreals[sessionid].flush_talk()

    if params['type']=='echo':
        nerfreals[sessionid].put_msg_txt(params['text'])
    elif params['type']=='chat':
        pass
        # TODO: 
        # res=await asyncio.get_event_loop().run_in_executor(None, llm_response, params['text'],nerfreals[sessionid])
        #nerfreals[sessionid].put_msg_txt(res)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data":"ok"}
        ),
    )

async def humanaudio(request):
    try:
        form= await request.post()
        sessionid = int(form.get('sessionid',0))
        fileobj = form["file"]
        filename=fileobj.filename
        filebytes=fileobj.file.read()
        nerfreals[sessionid].put_audio_file(filebytes)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg":"err","data": ""+e.args[0]+""}
            ),
        )

async def set_audiotype(request):
    params = await request.json()

    sessionid = params.get('sessionid',0)    
    nerfreals[sessionid].set_curr_state(params['audiotype'],params['reinit'])

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data":"ok"}
        ),
    )

async def record(request):
    params = await request.json()

    sessionid = params.get('sessionid',0)
    if params['type']=='start_record':
        # nerfreals[sessionid].put_msg_txt(params['text'])
        nerfreals[sessionid].start_recording()
    elif params['type']=='end_record':
        nerfreals[sessionid].stop_recording()
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data":"ok"}
        ),
    )

async def is_speaking(request):
    params = await request.json()

    sessionid = params.get('sessionid',0)
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data": nerfreals[sessionid].is_speaking()}
        ),
    )

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

async def post(url,data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url,data=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        print(f'Error: {e}')

async def run(push_url,sessionid):
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid)
    nerfreals[sessionid] = nerfreal

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url,pc.localDescription.sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer,type='answer'))
##########################################
# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MULTIPROCESSING_METHOD'] = 'forkserver'                                                    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_device', type=str, default='0') # 0,1,2,3,...

    parser.add_argument('--avatar', type=str, default='ernerf') #ernerf musetalk wav2lip

    parser.add_argument('--fps', type=int, default=50) # audio FPS

    parser.add_argument('--tts', type=str, default='edgetts') #xtts gpt-sovits cosyvoice

    parser.add_argument('--transport', type=str, default='webrtc') #rtmp webrtc rtcpush
    parser.add_argument('--push_url', type=str, default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream') #rtmp://localhost/live/livestream

    parser.add_argument('--max_session', type=int, default=1)  #multi session count
    parser.add_argument('--listenport', type=int, default=8010)

    opt = parser.parse_args()

    # TODO: Use TalkReal, load model and avatar
    if opt.model == 'ernerf': 
        from nerfreal import NeRFReal,load_model,load_avatar
        model = load_model(opt)
        avatar = load_avatar(opt) 
    elif opt.model == 'musetalk':
        from musereal import MuseReal,load_model,load_avatar,warm_up
        print(opt)
        model = load_model()
        avatar = load_avatar(opt.avatar_id) 
        warm_up(opt.batch_size,model)      
    elif opt.model == 'wav2lip':
        from lipreal import LipReal,load_model,load_avatar,warm_up
        print(opt)
        model = load_model("./models/wav2lip.pth")
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,model,96)

    #############################################################################
    appasync = web.Application()
    appasync.on_shutdown.append(on_shutdown)
    appasync.router.add_post("/offer", offer)
    appasync.router.add_post("/human", human)
    appasync.router.add_post("/humanaudio", humanaudio)
    appasync.router.add_post("/set_audiotype", set_audiotype)
    appasync.router.add_post("/record", record)
    appasync.router.add_post("/is_speaking", is_speaking)
    appasync.router.add_static('/',path='web')

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(appasync, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
    # Configure CORS on all routes.
    for route in list(appasync.router.routes()):
        cors.add(route)

    pagename='webrtcapi.html'
    if opt.transport=='rtmp':
        pagename='echoapi.html'
    elif opt.transport=='rtcpush':
        pagename='rtcpushapi.html'
    print('start http server; http://<serverip>:'+str(opt.listenport)+'/'+pagename)
    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())
        if opt.transport=='rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k!=0:
                    push_url = opt.push_url+str(k)
                loop.run_until_complete(run(push_url,k))
        loop.run_forever()    

    run_server(web.AppRunner(appasync))
