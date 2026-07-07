import re
from typing import Callable, List, Optional


_HARD_BOUNDARY_CHARS = set("。！？!?；;：:\n")
_SOFT_BOUNDARY_CHARS = set("，,、)]}】》」』\"' \t")
_SENTENCE_BOUNDARY_CHARS = set("，,。！？!?；;：:\n")


def _has_unclosed_spk_tag(text: str) -> bool:
    return text.rfind("<|") > text.rfind("|>")


def _has_unclosed_square_special(text: str) -> bool:
    left = text.rfind("[")
    right = text.rfind("]")
    if left <= right:
        return False
    frag = text[left:]
    return bool(re.fullmatch(r"\[[A-Za-z0-9_/\-]*", frag))


def _has_unclosed_angle_special(text: str) -> bool:
    left = text.rfind("<")
    right = text.rfind(">")
    if left <= right:
        return False
    if text.startswith("<|", left):
        return False
    frag = text[left:]
    return bool(re.fullmatch(r"</?[A-Za-z0-9_\-]*", frag))


def _has_unclosed_special(text: str) -> bool:
    return (
        _has_unclosed_spk_tag(text)
        or _has_unclosed_square_special(text)
        or _has_unclosed_angle_special(text)
    )


def _ends_in_numeric_tail(text: str) -> bool:
    tail = text[-32:]
    if not tail:
        return False
    if re.search(r"\d$", tail):
        return True
    if re.search(r"(?:\d[\d\s]*[.\-:/])$", tail):
        return True
    if re.search(r"(?:\d[\d\s.\-:/]*[A-Za-z])$", tail):
        return True
    if re.search(r"(?:\d[\d\s.\-:/]*[年月日号点分秒元块兆千百十万亿%])$", tail):
        return True
    return False


def _is_numeric_release(text: str) -> bool:
    return len(text) >= 2 and _ends_in_numeric_tail(text[:-1]) and not _ends_in_numeric_tail(text)


def _is_special_release(text: str) -> bool:
    return len(text) >= 2 and _has_unclosed_special(text[:-1]) and not _has_unclosed_special(text)


class StreamingTextProcessor:
    """Chunk raw text into online-safe pieces for CosyVoice2 bi-stream inference."""

    def __init__(
        self,
        tokenize: Callable[[str], List[int]],
        normalize: Callable[[str], str],
        min_chunk_tokens: int = 5,
        max_chunk_tokens: int = 20,
        first_chunk_tokens: Optional[int] = None,
        force_chunk_tokens: Optional[int] = None,
        spk_tag: str = "",
    ):
        self.tokenize = tokenize
        self.normalize = normalize
        self.min_chunk_tokens = max(1, int(min_chunk_tokens))
        self.max_chunk_tokens = max(self.min_chunk_tokens, int(max_chunk_tokens))
        if first_chunk_tokens is None:
            first_chunk_tokens = self.min_chunk_tokens
        self.first_chunk_tokens = max(self.min_chunk_tokens, int(first_chunk_tokens))
        if force_chunk_tokens is None:
            force_chunk_tokens = self.max_chunk_tokens + max(4, self.max_chunk_tokens // 2)
        self.force_chunk_tokens = max(self.max_chunk_tokens, int(force_chunk_tokens))
        self.spk_tag = spk_tag
        self.buffer = ""
        self.started = False

    def feed(self, text: str) -> List[str]:
        if text:
            self.buffer += text
        return self._drain(final=False)

    def flush(self) -> List[str]:
        return self._drain(final=True)

    def _target_tokens(self) -> int:
        return self.first_chunk_tokens if not self.started else self.max_chunk_tokens

    def _token_len(self, text: str) -> int:
        text = text.strip()
        if not text:
            return 0
        return len(self.tokenize(text))

    def _classify_boundary(self, prefix: str) -> Optional[str]:
        if not prefix.strip():
            return None
        if _has_unclosed_special(prefix):
            return None
        if _ends_in_numeric_tail(prefix):
            return None
        last = prefix[-1]
        if last in _HARD_BOUNDARY_CHARS:
            return "hard"
        if last in _SOFT_BOUNDARY_CHARS:
            return "soft"
        return None

    def _normalize_chunk(self, chunk: str) -> str:
        text = self.normalize(chunk).strip()
        if self.spk_tag and not self.started and text and not text.startswith("<|spk_"):
            text = f"{self.spk_tag}{text}"
        return text

    def _drain(self, final: bool) -> List[str]:
        out: List[str] = []
        while True:
            end = self._choose_commit_index(final=final)
            if end is None:
                break
            chunk = self.buffer[:end]
            self.buffer = self.buffer[end:]
            text = self._normalize_chunk(chunk)
            if text:
                out.append(text)
                self.started = True
            if not self.buffer:
                break
            if final:
                continue
        return out

    def _choose_commit_index(self, final: bool) -> Optional[int]:
        if not self.buffer:
            return None
        if final:
            return len(self.buffer)

        target_tokens = self._target_tokens()
        if self._token_len(self.buffer) < self.min_chunk_tokens:
            return None

        limit = len(self.buffer) - 1
        boundary_candidates = []
        safe_candidates = []

        for end in range(1, limit + 1):
            prefix = self.buffer[:end]
            token_len = self._token_len(prefix)
            if token_len < self.min_chunk_tokens:
                continue
            if _has_unclosed_special(prefix) or _ends_in_numeric_tail(prefix):
                continue
            if _is_numeric_release(prefix) or _is_special_release(prefix):
                safe_candidates.append((end, token_len))
                continue
            safe_candidates.append((end, token_len))
            kind = self._classify_boundary(prefix)
            if kind is not None:
                boundary_candidates.append((end, token_len, kind))

        if not safe_candidates:
            return None

        preferred = [c for c in boundary_candidates if c[1] <= target_tokens]
        hard_preferred = [c for c in preferred if c[2] == "hard"]
        if hard_preferred:
            return max(hard_preferred, key=lambda x: x[1])[0]
        soft_preferred = [c for c in preferred if c[2] == "soft"]
        if soft_preferred:
            return max(soft_preferred, key=lambda x: x[1])[0]

        total_tokens = self._token_len(self.buffer)
        if total_tokens < target_tokens:
            return None

        hard_extended = [
            c for c in boundary_candidates
            if c[2] == "hard" and target_tokens < c[1] <= self.force_chunk_tokens
        ]
        if hard_extended:
            return min(hard_extended, key=lambda x: x[1])[0]

        soft_extended = [
            c for c in boundary_candidates
            if c[2] == "soft" and target_tokens < c[1] <= self.force_chunk_tokens
        ]
        if soft_extended:
            return min(soft_extended, key=lambda x: x[1])[0]

        if total_tokens >= self.force_chunk_tokens:
            forced = [c for c in safe_candidates if c[1] <= self.force_chunk_tokens]
            if forced:
                return max(forced, key=lambda x: x[1])[0]
            return safe_candidates[-1][0]

        return None


def iter_stream_text_chunks(
    text: str,
    tokenize: Callable[[str], List[int]],
    normalize: Callable[[str], str],
    min_chunk_tokens: int = 5,
    max_chunk_tokens: int = 20,
    first_chunk_tokens: Optional[int] = None,
    force_chunk_tokens: Optional[int] = None,
    spk_tag: str = "",
):
    processor = StreamingTextProcessor(
        tokenize=tokenize,
        normalize=normalize,
        min_chunk_tokens=min_chunk_tokens,
        max_chunk_tokens=max_chunk_tokens,
        first_chunk_tokens=first_chunk_tokens,
        force_chunk_tokens=force_chunk_tokens,
        spk_tag=spk_tag,
    )
    for ch in text:
        for chunk in processor.feed(ch):
            yield chunk
    for chunk in processor.flush():
        yield chunk


def split_text_by_punctuation(text: str) -> List[str]:
    parts: List[str] = []
    start = 0
    n = len(text)
    for i, ch in enumerate(text):
        if ch not in _SENTENCE_BOUNDARY_CHARS:
            continue
        prev_ch = text[i - 1] if i > 0 else ""
        next_ch = text[i + 1] if i + 1 < n else ""
        if ch in {",", ".", ":"} and prev_ch.isdigit() and next_ch.isdigit():
            continue
        end = i + 1
        if end < n and text[end] in {'"', "”", "'", "’", "」", "』"}:
            end += 1
        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
        start = end
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts
