# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Customer-service *pre*-TN (runs before wetext.Normalizer.normalize).
# wetext 本体用 kaldifst + ModelScope FST，改规则成本高；此处用与 wetext 相同的「先 shield 再交给下游」思路：
# 将客服场景中应「逐位 / 幺两三四」朗读的数字提前换成汉字，避免 FST 读成「六千七百八十九」等整数。
#
# 参考 wetext 包接口风格：wetext/wetext.py — Normalizer.normalize 在含数字时走 tag+verbalize。
#
# Enable:  export COSYVOICE_CUSTOMER_SERVICE_TN=1
# Disable: unset 或 COSYVOICE_CUSTOMER_SERVICE_TN=0

from __future__ import annotations

import os
import re
from typing import Callable, List, Pattern, Tuple

# 逐位读：1 读「一」（尾号、身份证后四位等）
_DIGIT_ZH = str.maketrans("0123456789", "零一二三四五六七八九")
# 电话 / 热线式：1 读「幺」
_DIGIT_ZH_YAO = str.maketrans("0123456789", "零幺二三四五六七八九")

# 非数字边界，减少订单号、金额中的子串误伤
_NB = r"(?<![0-9A-Za-z])"
_NE = r"(?![0-9A-Za-z])"


def _enabled() -> bool:
    v = os.environ.get("COSYVOICE_CUSTOMER_SERVICE_TN", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _serial_zh(digits: str, *, yao_one: bool) -> str:
    """将纯数字串转为汉字逐位读法。"""
    table = _DIGIT_ZH_YAO if yao_one else _DIGIT_ZH
    return digits.translate(table)


def _only_digits(s: str) -> str:
    return re.sub(r"[^0-9]", "", s)


def _int_to_zh_cardinal(n: int) -> str:
    """1–9999 整数转中文基数读法（三百、一千二百），用于带宽档位等。"""
    if n <= 0 or n > 9999:
        raise ValueError(n)
    cn = "零一二三四五六七八九"
    if n < 10:
        return cn[n]
    if n == 10:
        return "十"
    if n < 20:
        return "十" + cn[n % 10]
    if n < 100:
        t, o = divmod(n, 10)
        return cn[t] + "十" + (cn[o] if o else "")
    if n < 1000:
        h, r = divmod(n, 100)
        s = cn[h] + "百"
        if r == 0:
            return s
        if r < 10:
            return s + "零" + cn[r]
        return s + _int_to_zh_cardinal(r)
    th, r = divmod(n, 1000)
    s = cn[th] + "千"
    if r == 0:
        return s
    if r < 100:
        return s + "零" + _int_to_zh_cardinal(r)
    return s + _int_to_zh_cardinal(r)


def _apply_rules(text: str, rules: List[Tuple[Pattern[str], Callable[[re.Match], str]]]) -> str:
    for pat, fn in rules:
        text = pat.sub(fn, text)
    return text


# ---------------------------------------------------------------------------
# 规则表：顺序敏感 — 更长、更具体的模式在前
# ---------------------------------------------------------------------------

def _rules_broadband_rate() -> List[Tuple[Pattern[str], Callable[[re.Match], str]]]:
    """宽带 / 速率中的 M、G（兆、千兆），在 wetext 之前收口，避免 300 被读成「三零零」。

    wetext FST 对单独数字常做逐位读；带单位的「300M」应读「三百兆」。
    """

    # 排除 MHz、MB/s、Mbps 等（保留给 wetext 或原文）
    _not_measure = r"(?![Hh][Zz]|[Bb](?:/?[Ss])?(?:[Pp][Ss])?|[Pp][Xx])"

    def _m_repl(m: re.Match) -> str:
        raw = m.group(1)
        try:
            n = int(raw)
        except ValueError:
            return m.group(0)
        if n == 0:
            return m.group(0)
        try:
            zh = _int_to_zh_cardinal(n)
        except ValueError:
            return m.group(0)
        return zh + "兆"

    # 如 300M、300 m、300M的带宽、300M在…（后面跟汉字/标点/句读）
    m_pat = re.compile(
        r"(\d{1,4})\s*([Mm])"
        + _not_measure
        + r"(?=[\s,，.。!！?？:：、；在的是与或和|\|]|$|[\u4e00-\u9fff])"
    )
    m_rules: List[Tuple[Pattern[str], Callable[[re.Match], str]]] = [(m_pat, _m_repl)]

    def _g_repl(m: re.Match) -> str:
        raw = m.group(1)
        try:
            n = int(raw)
        except ValueError:
            return m.group(0)
        if n <= 0:
            return m.group(0)
        # 1G/2G 带宽口语：千兆、两千兆；避免处理「5G手机」类（G 后跟「手」「网」等）
        if n == 1:
            return "千兆"
        try:
            return _int_to_zh_cardinal(n) + "千兆"
        except ValueError:
            return m.group(0)

    # 仅当后面紧跟带宽相关词，避免误改「5G手机」
    g_pat = re.compile(
        r"(\d{1,2})\s*([Gg])"
        + _not_measure
        + r"(?=(?:的|是)?(?:带宽|光纤|宽带|套餐|速率|网络|网速|提速))"
    )

    m_rules.append((g_pat, _g_repl))
    return m_rules


def _rules_carrier_hotline() -> List[Tuple[Pattern[str], Callable[[re.Match], str]]]:
    """品牌 / 运营商 + 官方客服号（数字在模式中固定）。"""

    def _mk2(brand_re: str, num: str) -> Tuple[Pattern[str], Callable[[re.Match], str]]:
        pat = re.compile(f"({brand_re})([\\s：:,，]*?)({num}){_NE}")

        def _fn(m: re.Match) -> str:
            return m.group(1) + m.group(2) + _serial_zh(m.group(3), yao_one=True)

        return pat, _fn

    rs: List[Tuple[Pattern[str], Callable[[re.Match], str]]] = [
        _mk2(r"中国移动", "10086"),
        _mk2(r"(?:中国联通|联通(?![营业]))", "10010"),
        _mk2(r"(?:中国电信|电信(?![营业]))", "10000"),
        _mk2(r"中国广电", "10099"),
        _mk2(r"(?:工商银行|工行)", "95588"),
        _mk2(r"(?:建设银行|建行)", "95533"),
        _mk2(r"(?:农业银行|农行)", "95599"),
        _mk2(r"(?:中国银行|中行)", "95566"),
        _mk2(r"(?:交通银行|交行)", "95559"),
        _mk2(r"(?:招商银行|招行)", "95555"),
        _mk2(r"(?:邮储银行|邮储)", "95580"),
        _mk2(r"(?:光大银行|光大)", "95595"),
        _mk2(r"(?:民生银行|民生)", "95568"),
        _mk2(r"(?:兴业银行|兴业)", "95561"),
        _mk2(r"(?:中信银行|中信)", "95558"),
        _mk2(r"(?:浦发银行|浦发)", "95528"),
        _mk2(r"(?:平安银行|平安)", "95511"),
    ]
    return rs


def _rules_special_lines() -> List[Tuple[Pattern[str], Callable[[re.Match], str]]]:
    """400 / 800 / 95 短号 / 常见政府公益号等整块。"""

    rs: List[Tuple[Pattern[str], Callable[[re.Match], str]]] = []

    # 400-xxx-xxxx / 400xxxxxxx
    p400 = re.compile(_NB + r"(400[\s\-]?\d{3}[\s\-]?\d{4})" + _NE)

    def _f400(m: re.Match) -> str:
        d = _only_digits(m.group(1))
        return _serial_zh(d, yao_one=True) if len(d) == 10 else m.group(0)

    rs.append((p400, _f400))

    p800 = re.compile(_NB + r"(800[\s\-]?\d{3}[\s\-]?\d{4})" + _NE)

    def _f800(m: re.Match) -> str:
        d = _only_digits(m.group(1))
        return _serial_zh(d, yao_one=True) if len(d) == 10 else m.group(0)

    rs.append((p800, _f800))

    # 95xxx 银行 / 电信通用客服（5 位）
    p95 = re.compile(_NB + r"(95\d{3})" + _NE)

    def _f95(m: re.Match) -> str:
        return _serial_zh(m.group(1), yao_one=True)

    rs.append((p95, _f95))

    # 12345 市民热线、12315 等
    p12 = re.compile(_NB + r"(12[0-9]{3})" + _NE)

    def _f12(m: re.Match) -> str:
        return _serial_zh(m.group(1), yao_one=True)

    rs.append((p12, _f12))

    return rs


def _rules_mobile_11() -> List[Tuple[Pattern[str], Callable[[re.Match], str]]]:
    """11 位手机号：用语境约束，避免误匹配其它长数字。"""

    # 覆盖「您的手机号是」「本机号码为」等（「手机号」单独一条无法匹配带前缀的整句）
    ctx = (
        r"(?:(?:您的|来电|本机|联系|绑定|预留|登记|认证|验证)(?:手机号码?|电话(?:号码)?)|"
        r"(?:手机|电话)(?:号码)?|联系(?:方式|电话)|"
        r"(?:请(?:您)?)?(?:确认|核对)(?:一下)?(?:手机|电话|手机号码?)|"
        r"(?:号码|电话|手机号)(?:为|是|：|:))"
    )
    pat = re.compile(r"(" + ctx + r")([\s：:,，是为]*)(1[3-9]\d{9})" + _NE)

    def _fn(m: re.Match) -> str:
        return m.group(1) + m.group(2) + _serial_zh(m.group(3), yao_one=True)

    rs: List[Tuple[Pattern[str], Callable[[re.Match], str]]] = [(pat, _fn)]

    # 「您是 147… 号码的机主」类：号码在「号码」一词之前，若未 shield，wetext 会读成「五千两百一十三万…」等大数
    pat_you_number = re.compile(
        r"(您是|请问您是|跟您确认您是|麻烦您确认您是|跟您核对您是|"
        r"确认(?:一下)?[，,。.]?\s*您是|"
        r"核对(?:一下)?[，,。.]?\s*您是)"
        r"([\s：:,，]*)"
        r"(1[3-9]\d{9})"
        r"(\s*号码)"
    )

    def _fn_you(m: re.Match) -> str:
        return m.group(1) + m.group(2) + _serial_zh(m.group(3), yao_one=True) + m.group(4)

    rs.append((pat_you_number, _fn_you))
    return rs


def _rules_tail_and_id() -> List[Tuple[Pattern[str], Callable[[re.Match], str]]]:
    """尾号、后四位、身份证后四位等：逐位读，1 用「一」。"""

    rs: List[Tuple[Pattern[str], Callable[[re.Match], str]]] = []

    tail_pat = re.compile(
        r"((?:手机|电话|本机|注册|绑定|银行)?(?:卡)?尾号|"
        r"(?:银行卡|储蓄卡|信用卡|借记卡)?(?:号)?后[四4]位|"
        r"后[四4]位|末[四4]位|"
        r"身份证(?:号)?后[四4]位|证件(?:号)?后[四4]位|"
        r"账号后[四4]位|"
        r"卡号后[四4]位)"
        r"([\s：是为，,]*?)([0-9]{3,8})"
        + _NE
    )

    def _tail_fn(m: re.Match) -> str:
        return m.group(1) + m.group(2) + _serial_zh(m.group(3), yao_one=False)

    rs.append((tail_pat, _tail_fn))

    # 兼容「手机尾号 6789」已在 tail_pat 的「手机...尾号」中覆盖；单独「尾号」:
    simple = re.compile(r"((?:手机)?尾号)([\s：:,，]*)([0-9]{3,8})" + _NE)

    def _s_fn(m: re.Match) -> str:
        return m.group(1) + m.group(2) + _serial_zh(m.group(3), yao_one=False)

    rs.append((simple, _s_fn))
    return rs


def _rules_verify_order_ext() -> List[Tuple[Pattern[str], Callable[[re.Match], str]]]:
    """验证码、工单号、分机、邮编、IVR 按键。"""

    rs: List[Tuple[Pattern[str], Callable[[re.Match], str]]] = []

    vpat = re.compile(
        r"((?:短信)?验证码|动态(?:口令|密码)|校验码|"
        r"支付密码|查询密码|登录密码)((?:是|为|：|:|[\s，,])*)([0-9]{4,8})" + _NE
    )

    def _v_fn(m: re.Match) -> str:
        return m.group(1) + m.group(2) + _serial_zh(m.group(3), yao_one=True)

    rs.append((vpat, _v_fn))

    opat = re.compile(
        r"((?:工单|订单|案件|受理单|业务|参考)号(?:码)?)((?:是|为|：|:|[\s，,])*)([0-9]{6,22})" + _NE
    )

    def _o_fn(m: re.Match) -> str:
        return m.group(1) + m.group(2) + _serial_zh(m.group(3), yao_one=True)

    rs.append((opat, _o_fn))

    epat = re.compile(r"(分机(?:号)?)((?:是|为|：|:|[\s，,])*)([0-9]{2,5})" + _NE)

    def _e_fn(m: re.Match) -> str:
        # 分机一般与尾号一致，1 读「一」
        return m.group(1) + m.group(2) + _serial_zh(m.group(3), yao_one=False)

    rs.append((epat, _e_fn))

    zpat = re.compile(r"(邮政编码|邮编)((?:是|为|：|:|[\s，,])*)([0-9]{6})" + _NE)

    def _z_fn(m: re.Match) -> str:
        return m.group(1) + m.group(2) + _serial_zh(m.group(3), yao_one=False)

    rs.append((zpat, _z_fn))

    # 坐席工号 / 员工编号
    wpat = re.compile(
        r"((?:坐席|客服|工号|员工编号|服务专员))((?:是|为|：|:|[\s，,])*)([0-9]{3,8})" + _NE
    )

    def _w_fn(m: re.Match) -> str:
        return m.group(1) + m.group(2) + _serial_zh(m.group(3), yao_one=True)

    rs.append((wpat, _w_fn))

    # IVR：请按 1 / 按 9 号键
    ivr = re.compile(r"((?:请(?:您)?)?(?:按|再按|接下来按|选择))\s*([0-9\*#])(?:\s*号)?(?:键)?")

    def _map_key(c: str) -> str:
        if c == "*":
            return "星号"
        if c == "#":
            return "井号"
        # IVR 单键多读「一」「二」，与常见语音提示一致
        return _serial_zh(c, yao_one=False)

    def _i_fn(m: re.Match) -> str:
        return m.group(1) + _map_key(m.group(2))

    rs.append((ivr, _i_fn))

    return rs


def _rules_dial_phrase() -> List[Tuple[Pattern[str], Callable[[re.Match], str]]]:
    """拨打 / 致电 / 联系 + 号码（5–12 位数字）。"""

    pat = re.compile(
        r"((?:请(?:您)?)?(?:可(?:以)?)?(?:拨打|致电|联系|咨询|呼叫)(?:客服)?(?:热线)?(?:电话)?[\s：:,，]*)"
        r"([0-9]{5,12})"
        + _NE
    )

    def _fn(m: re.Match) -> str:
        d = m.group(2)
        # 避免把日期或金额片段读错：仅当长度属于常见热线 / 手机
        if len(d) not in (5, 6, 8, 10, 11, 12):
            return m.group(0)
        if len(d) == 11 and not d.startswith("1"):
            return m.group(0)
        return m.group(1) + _serial_zh(d, yao_one=True)

    return [(pat, _fn)]


def _rules_standalone_mobile_11() -> List[Tuple[Pattern[str], Callable[[re.Match], str]]]:
    """兜底：连续 11 位且符合本网手机号段 (1[3-9]…)，按电话号码逐位读（幺）。

    放在规则链末尾，避免抢在「尾号 / 工单号 / 宽带档位」等更具体规则之前匹配。
    """

    pat = re.compile(_NB + r"(1[3-9]\d{9})" + _NE)

    def _fn(m: re.Match) -> str:
        return _serial_zh(m.group(1), yao_one=True)

    return [(pat, _fn)]


def _all_rules() -> List[Tuple[Pattern[str], Callable[[re.Match], str]]]:
    return (
        _rules_broadband_rate()
        + _rules_carrier_hotline()
        + _rules_special_lines()
        + _rules_mobile_11()
        + _rules_tail_and_id()
        + _rules_verify_order_ext()
        + _rules_dial_phrase()
        + _rules_standalone_mobile_11()
    )


def apply_customer_service_tn(text: str) -> str:
    """
    客服场景预 TN。需在 wetext.normalize 之前调用。

    示例（COSYVOICE_CUSTOMER_SERVICE_TN=1）::
        手机尾号 6789 ... 中国移动 10086 ...
      ->
        手机尾号 六七八九 ... 中国移动 幺零零八六 ...
    """
    if not text or not _enabled():
        return text
    return _apply_rules(text, _all_rules())


if __name__ == "__main__":
    os.environ["COSYVOICE_CUSTOMER_SERVICE_TN"] = "1"
    _s = (
        "您好，请问是手机尾号 6789 的主号人张先生吗？这里是中国移动 10086 客户服务中心。"
        "验证码 123456，请拨打 95588 联系工行。您的手机号是 13812345678。"
        "请按 1 转人工，分机 8001。"
    )
    print(apply_customer_service_tn(_s))
