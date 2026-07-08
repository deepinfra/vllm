# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import count
from queue import Queue
from typing import Any
from urllib.parse import parse_qs, urlparse

import msgspec
import zmq

from vllm.config.kv_events import KVEventsConfig
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import ExternalBlockHash

logger = init_logger(__name__)


def _to_signed_i64(value: int | None) -> int | None:
    """Reinterpret a Python int into signed 64-bit range (two's complement).

    vLLM emits block hashes as unsigned 64-bit ints when
    ``VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=1``; the Dynamo local indexer keys on
    signed i64. This mirrors ``_to_signed_i64`` in the TRT-LLM tee so both
    engines feed the indexer an identical id space.
    """
    if value is None:
        return None
    if value >= 2**63:
        return value - 2**64
    if value < -(2**63):
        return ((value + 2**63) % 2**64) - 2**63
    return value


class EventBatch(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
):
    ts: float
    events: list[Any]
    data_parallel_rank: int | None = None


class KVCacheEvent(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag=True,
):
    """Base class for all KV cache-related events"""


MEDIUM_GPU = "GPU"


class BlockStored(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    parent_block_hash: ExternalBlockHash | None
    token_ids: list[int]
    block_size: int

    lora_id: int | None
    """Deprecated: use `lora_name` for KV block key hash.
    Retained for backward compatibility.
    """

    medium: str | None
    lora_name: str | None

    extra_keys: list[tuple[Any, ...] | None] | None = None
    """Extra keys used in block hash computation, one entry per block in
    block_hashes. Each entry contains MM identifiers, LoRA name, cache_salt,
    prompt embedding hashes, etc. for that specific block. Exposed for external
    KV cache consumers to reconstruct block hashes.
    """

    group_idx: int | None = None

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.block_hashes),
                self.parent_block_hash,
                tuple(self.token_ids),
                self.block_size,
                self.lora_id,
                self.medium,
                tuple(self.extra_keys) if self.extra_keys else None,
                self.group_idx,
            )
        )


class BlockRemoved(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    medium: str | None
    group_idx: int | None = None

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.block_hashes),
                self.medium,
                self.group_idx,
            )
        )


class AllBlocksCleared(KVCacheEvent):
    pass


class KVEventBatch(EventBatch):
    events: list[BlockStored | BlockRemoved | AllBlocksCleared]


class KVEventAggregator:
    """
    Aggregates KV events across multiple workers.
    Tracks how many times each event appears and returns only those
    that were emitted by all workers.
    """

    __slots__ = ("_event_counter", "_num_workers")

    def __init__(self, num_workers: int) -> None:
        if num_workers <= 0:
            raise ValueError("num_workers must be greater than zero.")
        self._event_counter: Counter[KVCacheEvent] = Counter()
        self._num_workers: int = num_workers

    def add_events(self, events: list[KVCacheEvent]) -> None:
        """
        Add events from a worker batch.

        :param events: List of KVCacheEvent objects.
        """
        if not isinstance(events, list):
            raise TypeError("events must be a list of KVCacheEvent.")
        self._event_counter.update(events)

    def get_common_events(self) -> list[KVCacheEvent]:
        """
        Return events that appeared in all workers.

        :return: List of events present in all workers.
        """
        return [
            event
            for event, count in self._event_counter.items()
            if count == self._num_workers
        ]

    def get_all_events(self) -> list[KVCacheEvent]:
        """
        Return all events for all workers.

        :return: List of events for all workers.
        """
        return list(self._event_counter.elements())

    def clear_events(self) -> None:
        """
        Clear all tracked events.
        """
        self._event_counter.clear()

    def increment_workers(self, count: int = 1) -> None:
        """
        Increment the number of workers contributing events.

        :param count: Number to increment the workers by.
        """
        if count <= 0:
            raise ValueError("count must be positive.")
        self._num_workers += count

    def reset_workers(self) -> None:
        """
        Reset the number of workers to 1.
        """
        self._num_workers = 1

    def get_number_of_workers(self) -> int:
        """
        Return the number of workers.

        :return: int number of workers.
        """
        return self._num_workers

    def __repr__(self) -> str:
        return (
            f"<KVEventAggregator workers={self._num_workers}, "
            f"events={len(self._event_counter)}>"
        )


class KVConnectorKVEvents(ABC):
    """
    Abstract base class for KV events.
    Acts as a container for KV events from the connector.
    """

    @abstractmethod
    def add_events(self, events: list[KVCacheEvent]) -> None:
        raise NotImplementedError

    @abstractmethod
    def aggregate(self) -> "KVConnectorKVEvents":
        raise NotImplementedError

    @abstractmethod
    def increment_workers(self, count: int = 1) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_all_events(self) -> list[KVCacheEvent]:
        raise NotImplementedError

    @abstractmethod
    def get_number_of_workers(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def clear_events(self) -> None:
        raise NotImplementedError

    def merge(self, other: "KVConnectorKVEvents") -> "KVConnectorKVEvents":
        self.add_events(other.get_all_events())
        return self


class EventPublisher(ABC):
    """Lightweight publisher for EventBatch batches with data parallelism
    support.

    In data parallel setups, each DP rank runs its own EventPublisher instance
    to avoid duplicate events and ensure proper event attribution:

    - Each DP rank creates a separate publisher
    - Publishers automatically annotate events with their data_parallel_rank
    - This allows consumers to distinguish events from different DP ranks

    The publisher is responsible for adding DP metadata since the scheduler
    operates independently of DP topology and shouldn't need DP awareness.
    """

    def __init__(self, data_parallel_rank: int = 0) -> None:
        self._data_parallel_rank = data_parallel_rank

    @abstractmethod
    def publish(self, events: EventBatch) -> None:
        """Emit events in order.

        Implementations should guarantee at-least-once delivery and
        monotonic ordering (e.g., via sequence numbers).
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the publisher."""


class NullEventPublisher(EventPublisher):
    """No-op implementation (default when disabled)."""

    def publish(self, events) -> None:
        return

    def shutdown(self) -> None:
        return


class ZmqEventPublisher(EventPublisher):
    """Reliable PUB/ROUTER publisher with an in-memory replay buffer.

    Spawns a separate thread to handle publishing from a queue.

    Parameters
    ----------
    endpoint:
        PUB address. Use `tcp://*:5557` to bind or `tcp://host:5557` to
        connect.
    replay_endpoint:
        Optional ROUTER address for replay requests. When given, subscribers can
        request missed batches by sending the starting sequence number as an
        8-byte big-endian integer.
    buffer_steps:
        Number of past batches to keep for replay.
    hwm:
        ZeroMQ high-water-mark for PUB socket.
    max_queue_size:
        Maximum number of events to buffer in memory.
    topic:
        Topic to publish events to.
    """

    SHUTDOWN_TIMEOUT: float = 1.0
    END_SEQ = (-1).to_bytes(8, "big", signed=True)

    def __init__(
        self,
        data_parallel_rank: int,
        endpoint: str = "tcp://*:5557",
        replay_endpoint: str | None = None,
        buffer_steps: int = 10_000,
        hwm: int = 100_000,
        max_queue_size: int = 100_000,
        topic: str = "",
        enable_local_indexer: bool = False,
        kv_recover_port: int | None = None,
        block_size: int | None = None,
        worker_id: int = 0,
        main_group_idx: int | None = None,
    ) -> None:
        # Storage
        super().__init__(data_parallel_rank)
        self._event_queue = Queue[EventBatch | None](maxsize=max_queue_size)
        self._buffer = deque[tuple[int, bytes]](maxlen=buffer_steps)

        # --- optional in-process recovery state (deepinfra) ----------------- #
        # When enabled, every published event is also fed into a Dynamo
        # LocalKvIndexer (radix tree + replay ring buffer + snapshot), keyed by
        # event_id == seq. A separate stdlib HTTP server (started below, in this
        # EngineCore process) serves GET /kv_recover?start=&end= so the global
        # indexer can recover missed events or pull a full snapshot -- the same
        # contract the TRT-LLM tee exposes. Turning this on ALSO switches the
        # publisher to one-event-per-batch sends so seq == event_id holds; the
        # default (disabled) path is byte-for-byte upstream behavior.
        self._indexer = None
        self._recover_server: ThreadingHTTPServer | None = None
        self._recover_thread: threading.Thread | None = None
        self._block_size = block_size
        self._feed_skipped = 0
        # Hybrid models emit each KV event once per KV-cache group, at
        # different block granularities, with hashes SHARED across groups
        # (all derived from one hash_block_size-granular list). Only the main
        # full-attention group is a valid prefix-cache signal; other groups'
        # removals carry colliding hashes that would erase its blocks from the
        # index. In indexer mode we therefore keep ONLY this group's events --
        # for both the published stream and the indexer feed (downstream Dynamo
        # ingest ignores group_idx, so filtering must happen here at the
        # source). None keeps every event (single-group models / older builds).
        self._main_group_idx = main_group_idx
        self._group_filtered = 0
        if enable_local_indexer:
            # Imported lazily so the default path never needs the compiled
            # wheel. Ships from github.com/deepinfra/kv-local-indexer; the image
            # pip-installs it from the public release URL (KV_WHEEL_URL).
            from kv_local_indexer import LocalIndexer

            if block_size is None:
                raise ValueError(
                    "enable_local_indexer requires block_size to be passed "
                    "(EventPublisherFactory.create must forward it)")
            self._indexer = LocalIndexer(worker_id, block_size, buffer_steps)
            logger.info(
                "KV local indexer enabled (worker_id=%s, block_size=%s, "
                "buffer_steps=%s, main_group_idx=%s)",
                worker_id, block_size, buffer_steps, main_group_idx)
            self._start_recover_server(kv_recover_port, data_parallel_rank)

        # ZMQ sockets
        self._ctx = zmq.Context.instance()
        self._pub: zmq.Socket | None = None
        self._replay: zmq.Socket | None = None
        self._dp_rank = data_parallel_rank

        self._endpoint = self.offset_endpoint_port(endpoint, self._dp_rank)
        self._replay_endpoint = self.offset_endpoint_port(
            replay_endpoint, self._dp_rank
        )
        self._hwm = hwm
        self._socket_setup()

        # Payload
        self._seq_gen = count()
        self._topic_bytes = topic.encode("utf-8")

        # Thread
        self._running = True
        logger.info("Starting ZMQ publisher thread")

        self._thread = threading.Thread(
            target=self._publisher_thread, daemon=True, name="zmq-publisher"
        )
        self._thread.start()

    def publish(self, events: EventBatch) -> None:
        if not self._running:
            raise RuntimeError("Publisher is closed")
        if events.data_parallel_rank is None:
            events.data_parallel_rank = self._data_parallel_rank
        self._event_queue.put(events)

    def shutdown(self) -> None:
        """Stop the publisher thread and clean up resources."""
        self._running = False
        self._event_queue.put_nowait(None)

        start = time.time()
        pending_items = True
        while pending_items and (time.time() - start < self.SHUTDOWN_TIMEOUT):
            pending_items = not self._event_queue.empty()
            if pending_items:
                time.sleep(0.1)

        if pending_items:
            logger.warning(
                "Warning: Queue still has %s items after %s seconds timeout",
                self._event_queue.qsize(),
                self.SHUTDOWN_TIMEOUT,
            )

        if self._thread.is_alive():
            self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)

        # Stop the recovery HTTP server and local indexer (deepinfra).
        if self._recover_server is not None:
            try:
                self._recover_server.shutdown()
                self._recover_server.server_close()
            except Exception as e:  # noqa: BLE001
                logger.warning("KV recover server shutdown failed: %s", e)
        if self._indexer is not None:
            try:
                self._indexer.shutdown()
            except Exception as e:  # noqa: BLE001
                logger.warning("KV local indexer shutdown failed: %s", e)

        # Clean up ZMQ resources
        try:
            if self._pub is not None:
                self._pub.close(linger=0)
            if self._replay is not None:
                self._replay.close(linger=0)
        finally:
            pass  # Do not terminate context; other sockets may use it

    # ------------------------------------------------------------------ #
    # Recovery (deepinfra): feed the local indexer + serve /kv_recover     #
    # ------------------------------------------------------------------ #
    def get_recovery_json(
        self, start: int | None = None, end: int | None = None
    ) -> str | None:
        """Return recovery events for ``[start, end]`` as a JSON string.

        ``start`` is the first ``event_id`` (== ZMQ ``seq``) the caller is
        missing; ``None`` requests a full snapshot. Result is the
        externally-tagged ``WorkerKvQueryResponse`` produced by the indexer.
        ``None`` when the local indexer is disabled. May be heavy (a full tree
        dump) and blocks in Rust, so it is served off the publisher thread by
        the HTTP server's worker threads.
        """
        if self._indexer is None:
            return None
        return self._indexer.get_events_json(start, end)

    def _start_recover_server(
        self, kv_recover_port: int | None, data_parallel_rank: int
    ) -> None:
        if kv_recover_port is None:
            logger.warning(
                "enable_local_indexer set but kv_recover_port is None; "
                "recovery HTTP server not started")
            return
        # Offset by DP rank so each rank's server gets its own port, matching
        # the ZMQ endpoint port offset convention above.
        port = kv_recover_port + data_parallel_rank
        publisher = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path.rstrip("/") != "/kv_recover":
                    self.send_error(404, "not found")
                    return
                qs = parse_qs(parsed.query)

                def _int(name: str) -> int | None:
                    vals = qs.get(name)
                    return int(vals[0]) if vals else None

                try:
                    body = publisher.get_recovery_json(_int("start"), _int("end"))
                except Exception as e:  # noqa: BLE001
                    self.send_error(500, f"kv_recover failed: {e}")
                    return
                if body is None:
                    self.send_error(404, "KV local indexer not enabled")
                    return
                payload = body.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: Any) -> None:  # silence access log
                pass

        self._recover_server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
        self._recover_thread = threading.Thread(
            target=self._recover_server.serve_forever,
            daemon=True,
            name="kv-recover-http",
        )
        self._recover_thread.start()
        logger.info("KV recover HTTP server listening on 0.0.0.0:%s", port)

    _FEED_SKIP_LOG_INTERVAL = 1000

    def _is_main_group(self, event: Any) -> bool:
        """True if this event belongs to the main-attention KV cache group.

        Events without a group_idx (AllBlocksCleared, single-group models,
        older builds) always pass. When no main group was configured
        (main_group_idx=None), everything passes.
        """
        if self._main_group_idx is None:
            return True
        group_idx = getattr(event, "group_idx", None)
        return group_idx is None or group_idx == self._main_group_idx

    def _feed_indexer(self, seq: int, event: Any, dp_rank: int) -> None:
        """Feed one already-published event into the local indexer (seq==id).

        Runs on the publisher thread in seq order. Robust by design: this rides
        next to the routing path, so a bad event must never stop publishing.
        The catch is broad on purpose -- the underlying Rust indexer can raise a
        ``pyo3_runtime.PanicException`` (a ``BaseException`` subclass, NOT caught
        by ``except Exception``), which would otherwise kill this thread. We
        re-raise only genuine interrupts.
        """
        if self._indexer is None:
            return
        try:
            if isinstance(event, BlockStored):
                # The indexer slices token_ids into per-block chunks of
                # block_size, so it requires len(token_ids) == n_blocks *
                # block_size. vLLM breaks that invariant for models with null
                # blocks (sliding-window attention, Mamba align mode): those
                # blocks' tokens stay in token_ids but their hashes are dropped
                # from block_hashes. Feeding such an event would mis-map tokens
                # to hashes (and panic on a trailing null block), so skip the
                # local-indexer feed for it. The event still went out on the ZMQ
                # wire intact. Dense models (e.g. DeepSeek) never hit this.
                n = len(event.block_hashes)
                bs = self._block_size or 0
                if bs <= 0 or len(event.token_ids) != n * bs:
                    self._feed_skipped += 1
                    if self._feed_skipped % self._FEED_SKIP_LOG_INTERVAL == 1:
                        logger.warning(
                            "KV local indexer: skipped %s misaligned BlockStored "
                            "events (token_ids=%s, blocks=%s, block_size=%s); "
                            "expected token_ids == blocks * block_size. This is "
                            "expected for null-block models (sliding-window / "
                            "Mamba) and those are not indexed.",
                            self._feed_skipped, len(event.token_ids), n, bs)
                    return
                block_hashes = [_to_signed_i64(int(h)) for h in event.block_hashes]
                parent = (
                    _to_signed_i64(int(event.parent_block_hash))
                    if event.parent_block_hash is not None
                    else None
                )
                self._indexer.apply_stored(
                    seq,
                    event.token_ids,
                    block_hashes,
                    parent,
                    dp_rank,
                    event.lora_name,
                )
            elif isinstance(event, BlockRemoved):
                block_hashes = [_to_signed_i64(int(h)) for h in event.block_hashes]
                self._indexer.apply_removed(seq, block_hashes, dp_rank)
            elif isinstance(event, AllBlocksCleared):
                self._indexer.apply_cleared(seq, dp_rank)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:  # noqa: BLE001 -- see docstring (PanicException)
            logger.warning("KV local indexer apply failed: %s", e)

    def _socket_setup(self) -> None:
        """Initialize sockets
        https://pyzmq.readthedocs.io/en/v19.0.0/morethanbindings.html#thread-safety
        """
        if self._pub is None:
            self._pub = self._ctx.socket(zmq.PUB)
            self._pub.set_hwm(self._hwm)
            # Heuristic: bind if wildcard / * present, else connect.
            # bind stable, connect volatile convention
            if self._endpoint is not None and (
                "*" in self._endpoint
                or "::" in self._endpoint
                or self._endpoint.startswith("ipc://")
                or self._endpoint.startswith("inproc://")
            ):
                self._pub.bind(self._endpoint)
            elif self._endpoint is not None:
                self._pub.connect(self._endpoint)

        # Set up replay socket: use ROUTER
        # 1) handles multiple REQ clients (identities)
        # 2) lets us send back one request → many replies (streamed events)
        # 3) works in our non‑blocking poll loop alongside PUB
        if self._replay_endpoint is not None:
            self._replay = self._ctx.socket(zmq.ROUTER)
            self._replay.bind(self._replay_endpoint)

    def _publisher_thread(self) -> None:
        """Background thread that processes the event queue."""
        self._pack = msgspec.msgpack.Encoder()

        assert self._pub is not None  # narrows type for mypy

        while self._running or self._event_queue.qsize() > 0:
            # --- replay (non-critical) ---------------------------------
            if self._replay is not None and self._replay.poll(0):
                try:
                    self._service_replay()
                except Exception as e:
                    logger.exception("Error in replay: %s", e)

            # --- main queue (critical) ---------------------------------
            try:
                event = self._event_queue.get(timeout=0.1)
                if event is None:
                    break  # Sentinel received, exit thread
            except queue.Empty:
                continue

            try:
                if self._indexer is not None:
                    # Recovery mode: emit ONE event per ZMQ batch so that
                    # seq == event_id, the invariant the local indexer and the
                    # global consumer's gap detection are built around. Each
                    # sub-event is published, buffered, and fed to the indexer
                    # under its own seq.
                    dp_rank = event.data_parallel_rank
                    for sub in event.events:
                        # Keep only the main-attention group's events; other
                        # groups are duplicates at different granularities
                        # whose shared hashes would corrupt the index (and
                        # any downstream consumer -- Dynamo ingest ignores
                        # group_idx). Dropped BEFORE taking a seq, so the
                        # published stream stays gap-free.
                        if not self._is_main_group(sub):
                            self._group_filtered += 1
                            if self._group_filtered % 1_000_000 == 0:
                                logger.debug(
                                    "KV events: filtered %s non-main-group "
                                    "events (main_group_idx=%s)",
                                    self._group_filtered,
                                    self._main_group_idx)
                            continue
                        seq = next(self._seq_gen)
                        single = type(event)(
                            ts=event.ts,
                            events=[sub],
                            data_parallel_rank=dp_rank,
                        )
                        payload = self._pack.encode(single)
                        seq_bytes = seq.to_bytes(8, "big")
                        self._pub.send_multipart(
                            (self._topic_bytes, seq_bytes, payload))
                        self._buffer.append((seq, payload))
                        self._feed_indexer(seq, sub, dp_rank or 0)
                    self._event_queue.task_done()
                else:
                    seq = next(self._seq_gen)

                    payload = self._pack.encode(event)
                    seq_bytes = seq.to_bytes(8, "big")
                    self._pub.send_multipart(
                        (self._topic_bytes, seq_bytes, payload))

                    self._buffer.append((seq, payload))
                    self._event_queue.task_done()

            except Exception as e:
                # Publishing failed;  back-off a bit to avoid a tight error loop
                logger.exception("Error in publisher thread: %s", e)
                time.sleep(0.1)

    def _service_replay(self) -> None:
        """If a replay request is waiting, send buffered batches."""
        assert self._replay is not None  # narrows type for mypy

        frame = self._replay.recv_multipart()
        if len(frame) != 3:
            logger.warning("Invalid replay request: %s", frame)
            return
        client_id, _, start_seq_bytes = frame
        start_seq = int.from_bytes(start_seq_bytes, "big")

        for seq, buf in self._buffer:
            if seq >= start_seq:
                # [identity, empty_delim, seq_bytes, payload]
                # (identity, empty_delim) are stripped off by the router
                # receiving payload is (seq_bytes, payload)
                self._replay.send_multipart(
                    (client_id, b"", seq.to_bytes(8, "big"), buf)
                )
        # Send end of sequence marker
        # receiving payload is (-1, b""")
        self._replay.send_multipart((client_id, b"", self.END_SEQ, b""))

    @staticmethod
    def offset_endpoint_port(
        endpoint: str | None, data_parallel_rank: int
    ) -> str | None:
        """Helper function to offset the port in an endpoint by
            the data parallel rank.

        Args:
            endpoint: The endpoint string
                (e.g., "tcp://*:5557" or "inproc://cache")
            data_parallel_rank: The data parallel rank to offset by

        Returns:
            The endpoint with the port offset by data_parallel_rank
                or suffix appended
        """
        # Do nothing if input is None or data_parallel_rank is 0
        if not endpoint or data_parallel_rank == 0:
            return endpoint

        if "inproc" in endpoint:
            return f"{endpoint}_dp{data_parallel_rank}"
        if "tcp" in endpoint:
            if endpoint and ":" in endpoint:
                # Get everything after the last colon (the port)
                last_colon_idx = endpoint.rfind(":")
                base_addr = endpoint[:last_colon_idx]
                base_port = int(endpoint[last_colon_idx + 1 :])
                new_port = base_port + data_parallel_rank
                return f"{base_addr}:{new_port}"
            return endpoint
        raise ValueError("Invalid endpoint: must contain 'inproc' or 'tcp'")


class EventPublisherFactory:
    _registry: dict[str, Callable[..., EventPublisher]] = {
        "null": NullEventPublisher,
        "zmq": ZmqEventPublisher,
    }

    @classmethod
    def register_publisher(cls, name: str, ctor: Callable[..., EventPublisher]) -> None:
        if name in cls._registry:
            raise KeyError(f"publisher '{name}' already registered")
        cls._registry[name] = ctor

    @classmethod
    def create(
        cls,
        config: KVEventsConfig | None,
        data_parallel_rank: int = 0,
        block_size: int | None = None,
        worker_id: int = 0,
        main_group_idx: int | None = None,
    ) -> EventPublisher:
        """Create publisher from a config mapping.

        ``block_size``, ``worker_id`` and ``main_group_idx`` are runtime values
        (not config fields); they are forwarded to the publisher so the optional
        local indexer can be built and non-main KV-cache-group events filtered.
        They are injected only for the zmq path (the null path returns early),
        so custom publishers that do not accept them are unaffected.
        """
        if (
            config is None
            or not config.enable_kv_cache_events
            or config.publisher == "null"
        ):
            return NullEventPublisher()

        config_dict = asdict(config)

        kind = config_dict.pop("publisher")
        config_dict.pop("enable_kv_cache_events")
        config_dict["block_size"] = block_size
        config_dict["worker_id"] = worker_id
        config_dict["main_group_idx"] = main_group_idx
        try:
            constructor = cls._registry[kind]
        except KeyError as exc:
            raise ValueError(f"Unknown event publisher '{kind}'") from exc
        return constructor(data_parallel_rank=data_parallel_rank, **config_dict)
