"""Run the active learning CLI with SIGTERM converted into a clean shutdown.

Slurm terminates a job that reaches its wall-clock limit by sending SIGTERM to
every process in the job's cgroup, then SIGKILL after ``KillWait`` seconds.
Python installs no SIGTERM handler by default, so the interpreter dies without
unwinding and without running ``atexit`` hooks. wandb buffers an offline run in
its ``wandb-core`` service and only commits the transaction log when that hook
fires, so an unhandled SIGTERM discards every metric logged during the run --
measured as a 7-byte ``run-*.wandb`` file versus 6.6 KB when the hook runs.

Installing :func:`signal.default_int_handler` for SIGTERM makes it raise
``KeyboardInterrupt`` exactly as SIGINT does. The interpreter then unwinds
normally, ``atexit`` runs, and the offline run is written out and stays
syncable with ``wandb sync``.

Usage mirrors the ``activelearning`` console script::

    python scripts/run_with_sigterm_flush.py config/my_experiment.yaml
"""

import signal
import sys


def main() -> None:
    """Install the SIGTERM handler, then delegate to the normal entry point."""
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    from activelearning.main import main as activelearning_main

    try:
        activelearning_main()
    except KeyboardInterrupt:
        # Reaching the wall clock is the expected end of a time-boxed run, not
        # a failure. Exit quietly so the flushed outputs are the visible result.
        print(
            "Interrupted (SIGTERM/SIGINT); flushing run outputs.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
