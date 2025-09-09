import asyncio
from typing import Optional, Tuple
import numpy as np

from winsdk.windows.media.capture import (
    MediaCapture, MediaCaptureInitializationSettings,
    MediaCaptureSharingMode, MediaCaptureMemoryPreference,
    StreamingCaptureMode, MediaStreamType
)
from winsdk.windows.media.capture.frames import (
    MediaFrameSourceGroup, MediaFrameSourceKind
)

from utils.image import sbmp_to_gray

class IRCapture:
    """
    - IR 프레임 캡처(loop)
    - IR Torch 제어(WinRT InfraredTorchControl 지원 장치 한정)
    """
    def __init__(self):
        self.cap: Optional[MediaCapture] = None
        self.reader = None
        self.task: Optional[asyncio.Task] = None
        self.last_gray: Optional[np.ndarray] = None
        self.running = False
        self._source_id: Optional[str] = None

    async def start(self) -> bool:
        await self.stop()
        groups = await MediaFrameSourceGroup.find_all_async()
        group, info = None, None
        for g in groups:
            for s in g.source_infos:
                if s.source_kind == MediaFrameSourceKind.INFRARED and s.media_stream_type in (
                    MediaStreamType.VIDEO_PREVIEW, MediaStreamType.VIDEO_RECORD
                ):
                    group, info = g, s
                    break
            if group:
                break
        if not group:
            return False

        self.cap = MediaCapture()
        settings = MediaCaptureInitializationSettings()
        settings.source_group = group
        settings.sharing_mode = MediaCaptureSharingMode.EXCLUSIVE_CONTROL
        settings.memory_preference = MediaCaptureMemoryPreference.CPU
        settings.streaming_capture_mode = StreamingCaptureMode.VIDEO
        await self.cap.initialize_async(settings)

        self._source_id = info.id
        src = self.cap.frame_sources[self._source_id]
        self.reader = await self.cap.create_frame_reader_async(src)
        await self.reader.start_async()

        self.running = True
        self.task = asyncio.create_task(self._pull_loop())
        return True

    async def _pull_loop(self):
        try:
            while self.running:
                fr = self.reader.try_acquire_latest_frame()
                if fr is None:
                    await asyncio.sleep(0); continue
                with fr:
                    vmf = fr.video_media_frame
                    if vmf is None or vmf.software_bitmap is None:
                        await asyncio.sleep(0); continue
                    self.last_gray = sbmp_to_gray(vmf.software_bitmap)
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass

    async def stop(self):
        self.running = False
        if self.task:
            self.task.cancel()
            try: await self.task
            except asyncio.CancelledError: pass
            self.task = None
        if self.reader:
            try: await self.reader.stop_async()
            except Exception: pass
            self.reader = None
        if self.cap:
            try: self.cap.close()
            except Exception: pass
            self.cap = None
        self.last_gray = None
        self._source_id = None

    # ------ Torch Control ------
    async def set_torch(self, enable: bool, power: Optional[int] = None) -> Tuple[bool, str]:
        """
        Returns: (ok, message)
        - 장치가 InfraredTorchControl 미지원이면 False
        """
        if not self.cap:
            return False, "캡처가 초기화되지 않았습니다."
        vdc = getattr(self.cap, "video_device_controller", None)
        torch = getattr(vdc, "infrared_torch_control", None) if vdc else None
        if not torch or not getattr(torch, "supported", False):
            return False, "WinRT 표준 IR 토치 제어 미지원"
        try:
            torch.enabled = bool(enable)
            msg = f"torch.enabled={torch.enabled}"
            if enable and power is not None:
                if getattr(torch, "power_supported", False):
                    torch.power = int(max(0, min(100, power)))
                    msg += f", power={torch.power}"
                else:
                    msg += ", power 조절 미지원"
            return True, msg
        except Exception as e:
            return False, f"토치 제어 실패: {e}"
