import io
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from cvlab.metrics import _quiet_coco_output


def test_cpu_metrics_do_not_swallow_gpu_training_progress(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    entered, release = threading.Event(), threading.Event()

    def metrics():
        with _quiet_coco_output():
            print("COCO debug output")
            entered.set()
            assert release.wait(5)
            print("COCO done")

    with ThreadPoolExecutor(max_workers=1) as executor:
        job = executor.submit(metrics)
        try:
            assert entered.wait(5)
            print("epoch=1 batch=250 mean_loss=0.9", flush=True)
        finally:
            release.set()
        job.result()
    assert stream.getvalue() == "epoch=1 batch=250 mean_loss=0.9\n"
    assert sys.stdout is stream


def test_nested_quiet_context_restores_stream_after_error(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    with pytest.raises(RuntimeError):
        with _quiet_coco_output():
            with _quiet_coco_output():
                print("hidden")
            raise RuntimeError("metrics failed")
    print("visible")
    assert sys.stdout is stream
    assert stream.getvalue() == "visible\n"


def test_overlapping_quiet_threads_do_not_restore_stdout_too_early(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]

    def worker(index):
        with _quiet_coco_output():
            entered[index].set()
            assert release[index].wait(5)
            print("hidden")

    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = [executor.submit(worker, index) for index in range(2)]
        try:
            assert all(event.wait(5) for event in entered)
            release[0].set()
            jobs[0].result()
            print("GPU still training")
        finally:
            release[0].set()
            release[1].set()
        jobs[1].result()
    assert stream.getvalue() == "GPU still training\n"
    assert sys.stdout is stream
