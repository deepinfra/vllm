# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import Literal

from vllm.config.utils import config


@config
class KVEventsConfig:
    """Configuration for KV event publishing."""

    enable_kv_cache_events: bool = False
    """If True, enable KV cache events for tracking block storage and removal.
    Events can be published externally by zmq using the event publisher config.
    """

    publisher: Literal["null", "zmq"] = None  # type: ignore[assignment]
    """The publisher to use for publishing kv events. Can be "null", "zmq".
    """

    endpoint: str = "tcp://*:5557"
    """The zmq endpoint to use for publishing kv events.
    """

    replay_endpoint: str | None = None
    """The zmq endpoint to use for replaying kv events.
    """

    buffer_steps: int = 10_000
    """The number of steps to cache for replay endpoint. Will only save
    events from the last N steps for the replay endpoint.
    """

    hwm: int = 100_000
    """The zmq high water mark for the event publisher. After queueing N events,
    events will start dropping if the consumer is not keeping up.
    """

    max_queue_size: int = 100_000
    """The maximum number of events to queue while waiting for publishing.
    """

    topic: str = ""
    """The topic to use for the event publisher. Consumers can subscribe to
    this topic to receive events.
    """

    enable_local_indexer: bool = False
    """If True, feed every published event into an in-process Dynamo
    LocalKvIndexer (radix tree + replay ring buffer + snapshot) and serve
    recovery over HTTP at ``kv_recover_port``. Requires the ``kv_local_indexer``
    wheel. Switches the publisher to one-event-per-batch sends so seq ==
    event_id. Replaces the ZMQ ROUTER replay path for recovery.
    """

    kv_recover_port: int | None = None
    """Base TCP port for the ``GET /kv_recover?start=&end=`` recovery HTTP
    server (offset by data_parallel_rank). Only used when
    ``enable_local_indexer`` is True.
    """

    def __post_init__(self):
        if self.publisher is None:
            self.publisher = "zmq" if self.enable_kv_cache_events else "null"
