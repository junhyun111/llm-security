from __future__ import annotations

from dataclasses import dataclass, field


_PORTABILITY_API_ALIASES = {
    # Juliet's std_testcase.h maps these portability macros to the platform API.
    # Keeping the mapping here lets the semantic analyzer reason about the source
    # before preprocessing it.
    "ACCESS": "access",
    "FOPEN": "fopen",
    "OPEN": "open",
    "STAT": "stat",
}


@dataclass(slots=True, frozen=True)
class MemoryCopySpec:
    dst_arg: int
    src_arg: int
    length_arg: int | None


@dataclass(slots=True, frozen=True)
class AllocationSpec:
    size_args: tuple[int, ...]
    nullable_return: bool


@dataclass(slots=True, frozen=True)
class SourceSpec:
    output_args: tuple[int, ...] = ()
    tainted_return: bool = False


@dataclass(slots=True, frozen=True)
class SinkSpec:
    input_args: tuple[int, ...]


@dataclass(slots=True, frozen=True)
class ReleaseSpec:
    pointer_arg: int


@dataclass(slots=True, frozen=True)
class SanitizerSpec:
    result: bool = False
    arguments: tuple[int, ...] = ()


@dataclass(slots=True)
class ApiCatalog:
    memory_copy: dict[str, MemoryCopySpec] = field(default_factory=dict)
    allocations: dict[str, AllocationSpec] = field(default_factory=dict)
    releases: dict[str, ReleaseSpec] = field(default_factory=dict)
    taint_sources: dict[str, SourceSpec] = field(default_factory=dict)
    taint_sinks: dict[str, SinkSpec] = field(default_factory=dict)
    sanitizers: dict[str, SanitizerSpec] = field(default_factory=dict)
    thread_spawn: set[str] = field(default_factory=set)
    lock_acquire: set[str] = field(default_factory=set)
    lock_release: set[str] = field(default_factory=set)
    toctou_checks: dict[str, tuple[int, ...]] = field(default_factory=dict)
    toctou_uses: dict[str, tuple[int, ...]] = field(default_factory=dict)

    @classmethod
    def default(cls) -> ApiCatalog:
        return cls(
            memory_copy={
                "memcpy": MemoryCopySpec(0, 1, 2),
                "memmove": MemoryCopySpec(0, 1, 2),
                "strncpy": MemoryCopySpec(0, 1, 2),
                "strcpy": MemoryCopySpec(0, 1, None),
            },
            allocations={
                "malloc": AllocationSpec((0,), True),
                "calloc": AllocationSpec((0, 1), True),
                "realloc": AllocationSpec((1,), True),
            },
            releases={"free": ReleaseSpec(0)},
            taint_sources={
                "read": SourceSpec((1,)),
                "recv": SourceSpec((1,)),
                "fread": SourceSpec((0,)),
                "getenv": SourceSpec((), True),
                "gethostbyaddr": SourceSpec((), True),
                "gethostbyname": SourceSpec((), True),
            },
            taint_sinks={
                "system": SinkSpec((0,)),
                "popen": SinkSpec((0,)),
                "open": SinkSpec((0,)),
                "fopen": SinkSpec((0,)),
                "printf": SinkSpec((0,)),
                "fprintf": SinkSpec((1,)),
                "strcmp": SinkSpec((0, 1)),
                "strncmp": SinkSpec((0, 1)),
            },
            thread_spawn={"pthread_create", "stdThreadCreate"},
            lock_acquire={"pthread_mutex_lock", "stdThreadLockAcquire"},
            lock_release={"pthread_mutex_unlock", "stdThreadLockRelease"},
            toctou_checks={"access": (0,), "stat": (0,), "lstat": (0,)},
            toctou_uses={
                "open": (0,),
                "fopen": (0,),
                "unlink": (0,),
                "rename": (0, 1),
            },
        )

    @staticmethod
    def canonical_name(callee: str) -> str:
        value = callee.strip()
        if "." in value:
            value = value.rsplit(".", 1)[-1]
        if "::" in value:
            value = value.rsplit("::", 1)[-1]
        return _PORTABILITY_API_ALIASES.get(value, value)
