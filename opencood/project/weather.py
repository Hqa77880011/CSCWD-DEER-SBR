"""Worker-independent weather/geometry scheduling; Snow is evaluation-only."""
import hashlib
import json


def stable_seed(*parts, stream="weather"):
    payload = json.dumps(["project_fix_v1", stream, *parts], separators=(",", ":"), ensure_ascii=False)
    return int.from_bytes(hashlib.blake2b(payload.encode(), digest_size=8).digest(), "big")


def weather_for(sample_id, agent_id, epoch, seed, track):
    if track not in ("C", "W"):
        raise ValueError("track must be C or W")
    if track == "C":
        return "normal"
    return "fog" if stable_seed(str(sample_id), str(agent_id), int(epoch), int(seed)) % 7 < 3 else "normal"


def augmentation_seed(sample_id, epoch, seed, stream="geometry"):
    """One seed per cooperative sample: all agents and clean/wet share geometry."""
    return stable_seed(str(sample_id), int(epoch), int(seed), stream=stream) % (2 ** 32)
