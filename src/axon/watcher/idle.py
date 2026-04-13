import asyncio
import logging
import threading
from datetime import datetime
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_last_activity_time: Optional[datetime] = None
_idle_loop: Optional[asyncio.AbstractEventLoop] = None
_main_loop: Optional[asyncio.AbstractEventLoop] = None
_running = False
_IDLE_THRESHOLD_SECONDS = 5  # change to 180 for production


def reset_activity() -> None:
    global _last_activity_time
    _last_activity_time = datetime.now()
    logger.debug("Activity timer reset")


def start_idle_monitor(
    callback: Callable[[list], Any],
    main_loop: Optional[asyncio.AbstractEventLoop] = None,
) -> None:
    global _callback, _running, _last_activity_time, _idle_loop, _main_loop

    if _running:
        logger.warning("Idle monitor already running")
        return

    _callback = callback
    _main_loop = main_loop
    _running = True
    _last_activity_time = datetime.now()

    def run_loop():
        global _idle_loop
        _idle_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_idle_loop)

        async def check_loop():
            while _running:
                await asyncio.sleep(2)
                if _last_activity_time is None:
                    continue
                idle_seconds = (datetime.now() - _last_activity_time).total_seconds()
                if idle_seconds >= _IDLE_THRESHOLD_SECONDS:
                    logger.info(f"Idle for {idle_seconds:.0f}s, triggering callback")
                    try:
                        from axon.watcher.monitor import get_recent_changes

                        changes = get_recent_changes()
                        if _callback:
                            if _main_loop is not None:
                                future = asyncio.run_coroutine_threadsafe(
                                    _callback(changes), _main_loop
                                )
                                future.result(timeout=30)
                            else:
                                _callback(changes)
                        reset_activity()
                    except Exception as e:
                        logger.error(f"Error in idle callback: {e}")

        _idle_loop.run_until_complete(check_loop())

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    logger.info(f"Idle monitor started ({_IDLE_THRESHOLD_SECONDS}s threshold)")


def stop_idle_monitor() -> None:
    global _running, _idle_loop

    if not _running:
        return

    _running = False

    if _idle_loop is not None:

        def cancel_all():
            for task in asyncio.all_tasks(_idle_loop):
                task.cancel()

        _idle_loop.call_soon_threadsafe(cancel_all)
        _idle_loop.call_soon_threadsafe(_idle_loop.stop)
        _idle_loop = None

    logger.info("Idle monitor stopped")
