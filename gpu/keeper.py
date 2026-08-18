"""
AutoResearcher GPU Keeper

Keeps cloud GPU instances alive by maintaining minimal GPU activity.
Many cloud platforms (e.g., Aliyun PAI-DSW) reclaim instances if
GPUs are idle for extended periods (typically 3 hours).

This is the basic open-source version. It holds a small tensor on
the designated GPU to prevent reclamation.
"""

import signal
import sys
import time
import logging

logger = logging.getLogger("autoresearcher.gpu.keeper")


class GPUKeeper:
    """Minimal GPU keep-alive for cloud instances.

    Allocates a small tensor on the target GPU and periodically
    performs a tiny operation to register activity.
    """

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self._running = True
        self._tensor = None

        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)

    def start(self, interval: int = 300):
        """Start the keep-alive loop.

        Args:
            interval: Seconds between activity pings (default: 5 min)
        """
        try:
            import torch
        except ImportError:
            logger.error("PyTorch not installed. Cannot run GPU keeper.")
            return

        if not torch.cuda.is_available():
            logger.error("CUDA not available.")
            return

        device = torch.device(f"cuda:{self.gpu_id}")
        logger.info(f"GPU Keeper starting on GPU {self.gpu_id}")

        # Allocate a tiny tensor (4KB)
        self._tensor = torch.zeros(1024, device=device, dtype=torch.float32)

        while self._running:
            # Minimal activity: in-place operation
            self._tensor.add_(1.0)
            self._tensor.zero_()
            time.sleep(interval)

        # Cleanup
        del self._tensor
        torch.cuda.empty_cache()
        logger.info("GPU Keeper stopped.")

    def _shutdown(self, signum, frame):
        logger.info(f"Received signal {signum}. Shutting down keeper.")
        self._running = False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPU Keep-Alive Daemon")
    parser.add_argument("--gpu", type=int, required=True, help="GPU ID to keep alive")
    parser.add_argument("--interval", type=int, default=300, help="Ping interval (seconds)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [GPU-Keeper] %(message)s")
    keeper = GPUKeeper(gpu_id=args.gpu)
    keeper.start(interval=args.interval)


if __name__ == "__main__":
    main()
