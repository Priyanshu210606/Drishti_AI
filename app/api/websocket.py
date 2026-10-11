import asyncio
import base64
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.models.loader import models_registry
from app.inference.pipeline import ContinuousInferencePipeline

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Real-time bidirectional WebSocket connection.
    Implements a decoupled producer-consumer pipeline with latest-frame buffering.
    Camera frames arriving faster than inference are dropped cleanly to preserve freshness.
    """
    await websocket.accept()
    pipeline = ContinuousInferencePipeline(models_registry)

    # Latest-frame single-slot queue
    frame_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    is_active = True

    # Background inference worker consuming from latest-frame buffer
    async def inference_worker():
        while is_active:
            try:
                frame_bytes = await frame_queue.get()
                # Run synchronous inference in threadpool so event loop remains responsive
                response = await asyncio.to_thread(pipeline.process_frame, frame_bytes)
                await websocket.send_text(response.model_dump_json())
                frame_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Catch unexpected frame decoding errors without dropping the socket
                pass

    worker_task = asyncio.create_task(inference_worker())

    try:
        while True:
            # Receive frame or control message from browser
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                frame_bytes = message["bytes"]
            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    msg_type = payload.get("type", "")

                    if msg_type == "frame" and "data" in payload:
                        raw_data = payload["data"]
                        if "," in raw_data:
                            raw_data = raw_data.split(",")[1]
                        frame_bytes = base64.b64decode(raw_data)
                    elif msg_type == "config":
                        mode = payload.get("speech_mode")
                        if mode in ("brief", "normal", "detailed"):
                            pipeline.speech_mode = mode
                        continue
                    elif msg_type == "reset":
                        pipeline.temporal_buffer.clear()
                        pipeline.speech_manager.reset()
                        continue
                    else:
                        continue
                except Exception:
                    continue
            else:
                continue

            # Latest-frame queue strategy: discard older unread frame if present
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                    frame_queue.task_done()
                except (asyncio.QueueEmpty, ValueError):
                    pass

            await frame_queue.put(frame_bytes)

    except WebSocketDisconnect:
        pass
    finally:
        is_active = False
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
