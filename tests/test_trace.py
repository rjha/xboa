from dataclasses import dataclass
import json
import logging
from pathlib import Path

from softmaxx.trace import Trace, Translator
from softmaxx.util import (
    format_list,
    format_map,
    format_maps,
    format_record,
    format_records,
)


@dataclass
class Profile:
    department: str
    location: str
    _ssn: str  # Should be excluded


@dataclass
class UserRecord:
    user_id: int
    username: str
    profile: Profile
    _password_hash: str  # Should be excluded


class BrokenRepr:
    def __init__(self) -> None:
        self.self_ref = self  # Circular reference

    def __repr__(self) -> str:
        raise RuntimeError("broken __repr__")


def run_util_tests(raw_data: dict, user_records: list[UserRecord]) -> None:
    print("=== PART 1: softmaxx.util Direct Tests ===")

    print("\n1. single dict (all keys)")
    print(format_map(raw_data["single_dict"]))

    print("\n2. single dict (filtered: id, status)")
    print(format_map(raw_data["single_dict"], keys=["id", "status"]))

    print("\n3. format mappings (all keys)")
    print(format_maps(raw_data["mappings"]))

    print("\n4. format mappings (filtered: id, role)")
    print(format_maps(raw_data["mappings"], keys=["id", "role"]))

    print("\n5. format single record (all keys, _private stripped)")
    print(format_record(user_records[0]))

    print("\n6. format single record (filtered: username)")
    print(format_record(user_records[0], keys=["username"]))

    print("\n7. format records (all keys, nested _ssn stripped)")
    print(format_records(user_records))

    print("\n8. format records (filtered top-level: user_id, username)")
    print(format_records(user_records, keys=["user_id", "username"]))

    print("\n9. format plain list")
    print(format_list(raw_data["list"]))

    print("\n10. format mixed list")
    print(format_list(raw_data["mixed_list"]))

    print("\n11. format list of records")
    print(format_list(user_records))


def run_trace_tests(raw_data: dict, user_records: list[UserRecord]) -> None:
    print("\n=== PART 2: Trace & Translator End-to-End Tests ===")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger = logging.getLogger("softmaxx.test")

    trace_on = Trace(logger=logger, enabled=True)
    trace_off = Trace(logger=logger, enabled=False)

    print("\n12. Trace disabled (should emit nothing)")
    trace_off.log("This should not appear", Trace.Type.MAP, raw_data["single_dict"])

    print("\n13. Trace enabled across all types")
    trace_on.log("Ping message only")
    trace_on.log("Single map", Trace.Type.MAP, raw_data["single_dict"], keys=["id", "service"])
    trace_on.log("Multiple maps", Trace.Type.MAPS, raw_data["mappings"], keys=["user", "active"])
    trace_on.log("Single record", Trace.Type.RECORD, user_records[1])
    trace_on.log("Multiple records", Trace.Type.RECORDS, user_records, keys=["user_id", "profile"])
    trace_on.log("Plain list", Trace.Type.LIST, raw_data["list"])


def run_error_guard_tests(raw_data: dict) -> None:
    print("\n=== PART 3: Contract Guard & Failsafe Tests ===")

    # 1. Passing a single str instead of Sequence[str] for keys
    try:
        format_map(raw_data["single_dict"], keys="id")  # type: ignore[arg-type]
    except TypeError as exc:
        print(f"14. Caught single-string keys error: {exc}")

    # 2. Passing keys to Trace.Type.LIST
    try:
        Translator.translate(Trace.Type.LIST, raw_data["list"], keys=["id"])
    except TypeError as exc:
        print(f"15. Caught keys-on-LIST error: {exc}")

    # 3. Circular reference + broken __repr__ failsafe
    print("16. Circular ref + broken __repr__ failsafe:")
    print(format_record(BrokenRepr()))


def main() -> None:
    data_path = Path(__file__).parent / "fixtures" / "trace_data.json"
    raw_data = json.loads(data_path.read_text(encoding="utf-8"))

    user_records = [
        UserRecord(
            user_id=u["user_id"],
            username=u["username"],
            _password_hash=u["_password_hash"],
            profile=Profile(**u["profile"]),
        )
        for u in raw_data["users"]
    ]

    run_util_tests(raw_data, user_records)
    run_trace_tests(raw_data, user_records)
    run_error_guard_tests(raw_data)


if __name__ == "__main__":
    main()
