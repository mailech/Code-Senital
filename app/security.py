import hmac
import hashlib
from typing import Tuple


def verify_signature(secret: str, timestamp: str, body: bytes, provided_sig: str) -> Tuple[bool, str]:
    message = f"{timestamp}.{body.decode('utf-8')}".encode("utf-8")
    expected = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    expected_sig = f"sha256={expected}"
    return hmac.compare_digest(expected_sig, provided_sig), expected_sig
