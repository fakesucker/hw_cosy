import json
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="客服语音对比打分",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Paths
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = DATA_DIR / "results.json"
DEFAULT_PAIR_FILE = DATA_DIR / "default_pair.json"
YUZHOU_DIR = PROJECT_DIR / "yuzhou_data"
TEXT_DIR = YUZHOU_DIR / "text"
DIALOGUE_FILE = TEXT_DIR / "kefu_scene_dialogue.jsonl"
FIXED_SCENE_COUNT = 5


def show_dataframe(df: pd.DataFrame, **kwargs) -> None:
    """Streamlit's st.dataframe uses PyArrow; fall back to a static table if it is missing."""
    try:
        st.dataframe(df, **kwargs)
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) != "pyarrow" and "pyarrow" not in str(exc):
            raise
        st.caption("安装 pyarrow 可启用可排序的交互式表格：`pip install pyarrow`")
        display = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df
        st.table(display)


def find_audio_root() -> Path | None:
    """
    优先使用 text_wav；若不存在则回退到 test_wav，兼容当前目录实际情况。
    """
    candidates = [
        YUZHOU_DIR / "text_wav",
        YUZHOU_DIR / "test_wav",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return None


AUDIO_ROOT = find_audio_root()

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>
    .sample-container {
        background-color: white;
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 10px 0;
    }
    .model-card {
        background-color: #f8f9fa;
        padding: 14px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 12px;
    }
    /* 打分按钮（st.radio）美化为胶囊按钮 */
    div[role="radiogroup"] {
        gap: 8px !important;
        flex-wrap: wrap;
    }
    div[role="radiogroup"] > label {
        background: #f5f7ff;
        border: 1px solid #dbe2ff;
        border-radius: 999px;
        padding: 6px 14px;
        transition: all 0.15s ease-in-out;
    }
    div[role="radiogroup"] > label:hover {
        border-color: #9ab0ff;
        background: #eef2ff;
        box-shadow: 0 1px 4px rgba(66, 99, 235, 0.16);
    }
    div[role="radiogroup"] > label:has(input:checked) {
        border-color: #4c6fff;
        background: #4c6fff;
        color: #ffffff;
        box-shadow: 0 2px 8px rgba(66, 99, 235, 0.35);
    }
    div[role="radiogroup"] > label:has(input:checked) p {
        color: #ffffff !important;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Data utils
# -----------------------------
def ensure_results_file():
    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)


def load_results():
    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def write_results(results):
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_default_pair():
    if not DEFAULT_PAIR_FILE.exists():
        return {}
    try:
        with open(DEFAULT_PAIR_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_default_pair(pair: dict):
    with open(DEFAULT_PAIR_FILE, "w", encoding="utf-8") as f:
        json.dump(pair, f, indent=2, ensure_ascii=False)


def pick_scenes_by_start(scene_list: list[str], start_index_1based: int, k: int = FIXED_SCENE_COUNT) -> list[str]:
    """
    按“起始序号”固定取 k 条（支持循环取）。
    start_index_1based: 从 1 开始计数。
    """
    if not scene_list:
        return []
    ordered = list(scene_list)
    n = len(ordered)
    if n <= k:
        return ordered
    start0 = max(0, min(start_index_1based - 1, n - 1))
    out = []
    for i in range(k):
        out.append(ordered[(start0 + i) % n])
    return out


def load_scene_dialogues():
    """
    从 jsonl 加载 scene -> dialogue 映射。
    """
    mapping = {}
    if not DIALOGUE_FILE.exists():
        return mapping

    with open(DIALOGUE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            scene = obj.get("scene")
            dialogue = obj.get("dialogue")
            if scene and dialogue:
                mapping[str(scene)] = str(dialogue)
    return mapping


def discover_model_epoch_tree(audio_root: Path):
    """
    从音频目录构建两级选择树：
    第一级：模型（通常是 顶层目录/模型目录）
    第二级：epoch
    """
    tree = {}
    if not audio_root or not audio_root.exists():
        return tree

    for wav in sorted(audio_root.rglob("*.wav")):
        rel = wav.relative_to(audio_root)
        if len(rel.parts) < 3:
            continue
        model_key = "/".join(rel.parts[:-2])
        epoch_key = rel.parts[-2]
        scene_id = wav.stem
        tree.setdefault(model_key, {}).setdefault(epoch_key, {})[scene_id] = wav

    return tree


def score_total(service_quality: int) -> float:
    # 现在仅保留“客服度”单指标，总分即客服度本身（保持 float 便于展示/聚合）
    return float(service_quality)


def is_valid_model_epoch(tree: dict, model_group: str, epoch: str) -> bool:
    return bool(model_group in tree and epoch in tree.get(model_group, {}))


def get_fallback_pair(tree: dict, model_keys: list[str]):
    """
    返回可用的两个不同 model+epoch 组合作为兜底默认值。
    """
    all_leaves = []
    for g in model_keys:
        for e in sorted(tree.get(g, {}).keys()):
            all_leaves.append((g, e))
    if len(all_leaves) < 2:
        return None
    return {
        "model_1_group": all_leaves[0][0],
        "model_1_epoch": all_leaves[0][1],
        "model_2_group": all_leaves[1][0],
        "model_2_epoch": all_leaves[1][1],
    }


def resolve_runtime_pair(tree: dict, model_keys: list[str], saved: dict):
    """
    根据已保存默认配置生成当前运行使用的 pair；若无效则自动兜底。
    """
    fallback = get_fallback_pair(tree, model_keys)
    if fallback is None:
        return {"ok": False}

    p = {
        "model_1_group": saved.get("model_1_group"),
        "model_1_epoch": saved.get("model_1_epoch"),
        "model_2_group": saved.get("model_2_group"),
        "model_2_epoch": saved.get("model_2_epoch"),
    }
    valid = (
        is_valid_model_epoch(tree, p["model_1_group"], p["model_1_epoch"])
        and is_valid_model_epoch(tree, p["model_2_group"], p["model_2_epoch"])
        and not (
            p["model_1_group"] == p["model_2_group"]
            and p["model_1_epoch"] == p["model_2_epoch"]
        )
    )
    if not valid:
        p = fallback

    p["ok"] = True
    return p


def build_pair_scene_candidates(
    tree: dict,
    model_1_group: str,
    model_1_epoch: str,
    model_2_group: str,
    model_2_epoch: str,
    dialogue_mapping: dict,
) -> list[str]:
    m1_map = tree.get(model_1_group, {}).get(model_1_epoch, {})
    m2_map = tree.get(model_2_group, {}).get(model_2_epoch, {})
    scenes = sorted(set(m1_map.keys()) & set(m2_map.keys()))
    if dialogue_mapping:
        scenes = [s for s in scenes if s in dialogue_mapping]
    return scenes


def is_new_format_result(record: dict) -> bool:
    if not isinstance(record, dict):
        return False
    if not record.get("scene_id"):
        return False
    models = record.get("models")
    if not isinstance(models, dict) or not models:
        return False
    # 现在只要求“客服度”；历史数据包含其它字段也不影响
    required_keys = {"service_quality"}
    for v in models.values():
        if not isinstance(v, dict):
            return False
        if not required_keys.issubset(v.keys()):
            return False
    return True


def cleanup_legacy_results():
    """
    清理旧版残留评测数据，只保留当前结构（单指标：客服度）。
    """
    results = load_results()
    cleaned = [r for r in results if is_new_format_result(r)]
    removed = len(results) - len(cleaned)
    if removed > 0:
        write_results(cleaned)
    return removed


def normalize_results(results):
    """
    仅展示当前新版结果结构（单指标：客服度）。
    """
    rows = []
    for r in results:
        models = r.get("models")
        if not isinstance(models, dict):
            continue
        for model_name, model_score in models.items():
            ms = model_score or {}
            rows.append(
                {
                    "Evaluator": r.get("evaluator"),
                    "Sample ID": r.get("sample_id"),
                    "Scene": r.get("scene_id"),
                    "Model": model_name,
                    "客服度": ms.get("service_quality"),
                    "Total Score": ms.get("total", ms.get("total_score")),
                    "Timestamp": r.get("date") or r.get("timestamp"),
                }
            )
    return rows


# -----------------------------
# Initialize
# -----------------------------
ensure_results_file()
dialogue_map = load_scene_dialogues()
removed_legacy_count = cleanup_legacy_results()
all_model_tree = discover_model_epoch_tree(AUDIO_ROOT) if AUDIO_ROOT else {}
all_model_keys = sorted(all_model_tree.keys())
saved_default_pair = load_default_pair()
runtime_pair = resolve_runtime_pair(all_model_tree, all_model_keys, saved_default_pair)
if runtime_pair.get("ok", False) and not DEFAULT_PAIR_FILE.exists():
    write_default_pair(
        {
            "model_1_group": runtime_pair["model_1_group"],
            "model_1_epoch": runtime_pair["model_1_epoch"],
            "model_2_group": runtime_pair["model_2_group"],
            "model_2_epoch": runtime_pair["model_2_epoch"],
            "scene_start_index": 1,
        }
    )

# -----------------------------
# Session state init
# -----------------------------
if "current_sample_id" not in st.session_state:
    st.session_state.current_sample_id = 1
if "sample_slider" not in st.session_state:
    st.session_state.sample_slider = 1
if "submitted_sample_id" not in st.session_state:
    st.session_state.submitted_sample_id = None
if "ab_assignments" not in st.session_state:
    st.session_state.ab_assignments = {}
if "staged_scores" not in st.session_state:
    st.session_state.staged_scores = {}
if "active_pair_signature" not in st.session_state:
    st.session_state.active_pair_signature = ""
if "last_render_scene_id" not in st.session_state:
    st.session_state.last_render_scene_id = ""


def sync_from_slider():
    st.session_state.current_sample_id = st.session_state.sample_slider


def go_next_sample(max_id: int):
    if st.session_state.current_sample_id < max_id:
        nxt = st.session_state.current_sample_id + 1
        st.session_state.current_sample_id = nxt
        st.session_state.submitted_sample_id = None


def go_prev_sample():
    if st.session_state.current_sample_id > 1:
        prv = st.session_state.current_sample_id - 1
        st.session_state.current_sample_id = prv
        st.session_state.submitted_sample_id = None


def get_ab_assignment(scene_id: str, model_leaf_1: str, model_leaf_2: str):
    pair = sorted([model_leaf_1, model_leaf_2])
    key = f"{scene_id}||{pair[0]}||{pair[1]}"
    if key not in st.session_state.ab_assignments:
        if random.random() < 0.5:
            st.session_state.ab_assignments[key] = {"A": pair[0], "B": pair[1]}
        else:
            st.session_state.ab_assignments[key] = {"A": pair[1], "B": pair[0]}
    return st.session_state.ab_assignments[key]


def stash_scene_score(
    scene_id: str,
    sample_id: int,
    dialogue_text: str,
    assignment: dict,
    score_a: dict,
    score_b: dict,
):
    st.session_state.staged_scores[scene_id] = {
        "sample_id": sample_id,
        "scene_id": scene_id,
        "dialogue": dialogue_text,
        "models": {
            assignment["A"]: score_a,
            assignment["B"]: score_b,
        },
    }


def get_slot_score_from_state(scene_id: str, slot: str):
    s_key = f"service__{scene_id}__{slot}"
    service_quality = int(st.session_state.get(s_key, 3))
    return {
        "service_quality": service_quality,
        "total": score_total(service_quality),
    }


def restore_scene_score_to_state(scene_id: str, assignment: dict):
    """
    当回到某个场景时，将已暂存分数回填到控件状态。
    """
    staged = st.session_state.staged_scores.get(scene_id)
    if not staged:
        return

    models = staged.get("models", {})
    model_a = assignment.get("A")
    model_b = assignment.get("B")
    score_a = models.get(model_a)
    score_b = models.get(model_b)

    def _fill(slot: str, score: dict | None):
        if not isinstance(score, dict):
            return
        s_key = f"service__{scene_id}__{slot}"
        st.session_state[s_key] = int(score.get("service_quality", 3))

    _fill("A", score_a)
    _fill("B", score_b)


def ensure_scene_widget_defaults(scene_id: str, assignment: dict):
    """
    仅在进入场景时初始化控件值：
    - 若有暂存，回填暂存
    - 否则默认 3 分
    """
    models = (st.session_state.staged_scores.get(scene_id) or {}).get("models", {})

    def _set(slot: str, model_leaf: str):
        s_key = f"service__{scene_id}__{slot}"

        staged = models.get(model_leaf) if isinstance(models, dict) else None
        default_s = int(staged.get("service_quality", 3)) if isinstance(staged, dict) else 3

        if s_key not in st.session_state:
            st.session_state[s_key] = default_s

    _set("A", assignment.get("A"))
    _set("B", assignment.get("B"))


# -----------------------------
# UI Layout
# -----------------------------
st.title("🎧 客服语音场景对比打分")

with st.sidebar:
    st.header("评测信息")
    evaluator_name = st.text_input("评测人", placeholder="请输入姓名")
    evaluator_name = evaluator_name.strip()
    is_admin_user = evaluator_name == "任夏明"

    st.divider()
    available_pages = ["场景评测"]
    if is_admin_user:
        available_pages.extend(["结果汇总", "数据导出"])
    page = st.radio("页面", available_pages)
    if not is_admin_user:
        st.caption("仅管理员可查看结果汇总与数据导出")

    st.divider()
    st.header("模型与 Epoch")

    if not runtime_pair.get("ok", False):
        st.error("可用音频模型目录不足 2 个，请检查 text_wav/test_wav。")
        model_1_group = ""
        model_1_epoch = ""
        model_2_group = ""
        model_2_epoch = ""
    elif is_admin_user:
        st.caption("管理员可修改默认对比组合")
        d_m1g = runtime_pair["model_1_group"]
        d_m1e = runtime_pair["model_1_epoch"]
        d_m2g = runtime_pair["model_2_group"]
        d_m2e = runtime_pair["model_2_epoch"]
        d_start_idx = int(saved_default_pair.get("scene_start_index", 1)) if isinstance(saved_default_pair, dict) else 1

        st.markdown("**默认模型 1**")
        idx_m1g = all_model_keys.index(d_m1g) if d_m1g in all_model_keys else 0
        model_1_group = st.selectbox("第一步：选择模型", all_model_keys, index=idx_m1g, key="m1_group")
        m1_epochs = sorted(all_model_tree.get(model_1_group, {}).keys())
        idx_m1e = m1_epochs.index(d_m1e) if d_m1e in m1_epochs else 0
        model_1_epoch = st.selectbox("第二步：选择 epoch", m1_epochs, index=idx_m1e, key="m1_epoch")

        st.markdown("**默认模型 2**")
        idx_m2g = all_model_keys.index(d_m2g) if d_m2g in all_model_keys else min(1, len(all_model_keys) - 1)
        model_2_group = st.selectbox("第一步：选择模型 ", all_model_keys, index=idx_m2g, key="m2_group")
        m2_epochs = sorted(all_model_tree.get(model_2_group, {}).keys())
        idx_m2e = m2_epochs.index(d_m2e) if d_m2e in m2_epochs else 0
        model_2_epoch = st.selectbox("第二步：选择 epoch ", m2_epochs, index=idx_m2e, key="m2_epoch")

        model_1_leaf = f"{model_1_group}/{model_1_epoch}"
        model_2_leaf = f"{model_2_group}/{model_2_epoch}"
        if model_1_leaf == model_2_leaf:
            st.warning("模型 1 与 模型 2 不能指向同一个模型+epoch。")
        scene_order_for_index = sorted(dialogue_map.keys()) if dialogue_map else sorted(
            set(all_model_tree.get(model_1_group, {}).get(model_1_epoch, {}).keys())
            & set(all_model_tree.get(model_2_group, {}).get(model_2_epoch, {}).keys())
        )
        max_start = max(1, len(scene_order_for_index))
        scene_start_index = st.number_input(
            "固定场景起始序号（从1开始）",
            min_value=1,
            max_value=max_start,
            value=min(max(1, d_start_idx), max_start),
            step=1,
            help="系统会从该序号开始连续取5条（循环取）。",
        )
        if st.button("保存为默认组合", width="stretch"):
            if model_1_leaf == model_2_leaf:
                st.error("保存失败：两个默认组合不能相同。")
            else:
                write_default_pair(
                    {
                        "model_1_group": model_1_group,
                        "model_1_epoch": model_1_epoch,
                        "model_2_group": model_2_group,
                        "model_2_epoch": model_2_epoch,
                        "scene_start_index": int(scene_start_index),
                    }
                )
                st.success("已保存默认组合与固定场景起始序号。")
                st.rerun()
    else:
        model_1_group = runtime_pair["model_1_group"]
        model_1_epoch = runtime_pair["model_1_epoch"]
        model_2_group = runtime_pair["model_2_group"]
        model_2_epoch = runtime_pair["model_2_epoch"]

if all_model_keys:
    model_1_leaf = f"{model_1_group}/{model_1_epoch}" if model_1_group and model_1_epoch else ""
    model_2_leaf = f"{model_2_group}/{model_2_epoch}" if model_2_group and model_2_epoch else ""
    model_1_scene_map = all_model_tree.get(model_1_group, {}).get(model_1_epoch, {})
    model_2_scene_map = all_model_tree.get(model_2_group, {}).get(model_2_epoch, {})
else:
    model_1_leaf = ""
    model_2_leaf = ""
    model_1_scene_map = {}
    model_2_scene_map = {}

pair_signature = ""
if model_1_leaf and model_2_leaf and model_1_leaf != model_2_leaf:
    pair_signature = "||".join(sorted([model_1_leaf, model_2_leaf]))
    if st.session_state.active_pair_signature != pair_signature:
        st.session_state.active_pair_signature = pair_signature
        st.session_state.staged_scores = {}
        st.session_state.current_sample_id = 1
        st.session_state.sample_slider = 1

# 固定场景集合：按“全局场景序号”取固定5条，再过滤为当前模型组合可用
if dialogue_map:
    base_scene_order = sorted(dialogue_map.keys())
else:
    base_scene_order = build_pair_scene_candidates(
        all_model_tree,
        model_1_group,
        model_1_epoch,
        model_2_group,
        model_2_epoch,
        dialogue_map,
    ) if (model_1_leaf and model_2_leaf and model_1_leaf != model_2_leaf) else []

scene_start_index = int(saved_default_pair.get("scene_start_index", 1)) if isinstance(saved_default_pair, dict) else 1
fixed_scene_window = pick_scenes_by_start(base_scene_order, scene_start_index, FIXED_SCENE_COUNT)

pair_available_scenes = set(
    build_pair_scene_candidates(
        all_model_tree,
        model_1_group,
        model_1_epoch,
        model_2_group,
        model_2_epoch,
        dialogue_map,
    )
) if (model_1_leaf and model_2_leaf and model_1_leaf != model_2_leaf) else set()

selected_scenes = [s for s in fixed_scene_window if s in pair_available_scenes]

if selected_scenes:
    max_scene_id = len(selected_scenes)
    if st.session_state.current_sample_id > max_scene_id:
        st.session_state.current_sample_id = 1
        st.session_state.sample_slider = 1


# -----------------------------
# Evaluation Page
# -----------------------------
if page == "场景评测":
    st.header("场景评测")

    if not AUDIO_ROOT:
        st.error("未找到音频目录。请检查 `yuzhou_data/text_wav` 或 `yuzhou_data/test_wav`。")
        st.stop()
    if not DIALOGUE_FILE.exists():
        st.error(f"未找到场景文本文件：{DIALOGUE_FILE}")
        st.stop()
    if not evaluator_name:
        st.warning("请先在左侧填写评测人。")
        st.stop()
    if len(all_model_keys) < 2 or model_1_leaf == model_2_leaf:
        st.warning("请在左侧完成两个不同的“模型+epoch”选择。")
        st.stop()
    if not selected_scenes:
        st.error("当前默认序号对应的场景在该模型组合下不可用，请让管理员调整起始序号。")
        st.stop()

    # 先同步 slider 状态，避免组件实例化后再写 key 导致报错
    if st.session_state.sample_slider != st.session_state.current_sample_id:
        st.session_state.sample_slider = st.session_state.current_sample_id

    # 场景选择
    col_sel, col_prev, col_next = st.columns([3, 1, 1])
    with col_sel:
        st.slider(
            "选择场景",
            min_value=1,
            max_value=len(selected_scenes),
            key="sample_slider",
            on_change=sync_from_slider,
        )
    sample_id = st.session_state.current_sample_id
    scene_id = selected_scenes[sample_id - 1]
    dialogue_text = dialogue_map.get(scene_id, "")
    ab_assignment = get_ab_assignment(scene_id, model_1_leaf, model_2_leaf)
    # 只在“切换到新场景”时做一次回填/初始化，避免点击打分后被旧值覆盖
    if st.session_state.last_render_scene_id != scene_id:
        ensure_scene_widget_defaults(scene_id, ab_assignment)
        st.session_state.last_render_scene_id = scene_id

    with col_prev:
        st.write("")
        st.write("")
        if st.button("◀ 上一条", width="stretch"):
            score_a_nav = get_slot_score_from_state(scene_id, "A")
            score_b_nav = get_slot_score_from_state(scene_id, "B")
            stash_scene_score(scene_id, sample_id, dialogue_text, ab_assignment, score_a_nav, score_b_nav)
            go_prev_sample()
            st.rerun()
    with col_next:
        st.write("")
        st.write("")
        if st.button("下一条 ▶", width="stretch"):
            score_a_nav = get_slot_score_from_state(scene_id, "A")
            score_b_nav = get_slot_score_from_state(scene_id, "B")
            stash_scene_score(scene_id, sample_id, dialogue_text, ab_assignment, score_a_nav, score_b_nav)
            go_next_sample(len(selected_scenes))
            st.rerun()
    model_a_leaf = ab_assignment["A"]
    model_b_leaf = ab_assignment["B"]

    # 由 scene_map 反查到 wav 路径
    scene_to_path = {}
    scene_to_path.update(model_1_scene_map)
    scene_to_path.update(model_2_scene_map)

    st.markdown('<div class="sample-container">', unsafe_allow_html=True)
    st.subheader(f"场景 #{sample_id} · `{scene_id}`")
    st.markdown("**场景对话文本：**")
    st.text_area("dialogue", value=dialogue_text, height=260, disabled=True, label_visibility="collapsed")
    if is_admin_user:
        st.caption(f"A={model_a_leaf}")
        st.caption(f"B={model_b_leaf}")
    st.markdown("</div>", unsafe_allow_html=True)

    wav_a = scene_to_path[scene_id] if model_a_leaf in (model_1_leaf, model_2_leaf) else None
    wav_b = scene_to_path[scene_id] if model_b_leaf in (model_1_leaf, model_2_leaf) else None
    # 更精确按模型查，避免同 scene 覆盖
    if model_a_leaf == model_1_leaf:
        wav_a = model_1_scene_map[scene_id]
    elif model_a_leaf == model_2_leaf:
        wav_a = model_2_scene_map[scene_id]
    if model_b_leaf == model_1_leaf:
        wav_b = model_1_scene_map[scene_id]
    elif model_b_leaf == model_2_leaf:
        wav_b = model_2_scene_map[scene_id]

    st.subheader("音频 A/B 对比")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        st.markdown("**音频 A**")
        if is_admin_user:
            st.caption(model_a_leaf)
        st.audio(str(wav_a), format="audio/wav")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        st.markdown("**音频 B**")
        if is_admin_user:
            st.caption(model_b_leaf)
        st.audio(str(wav_b), format="audio/wav")
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("打分（1-5 按钮）")

    def render_model_score(panel_title: str, key_prefix: str):
        st.markdown(f"#### {panel_title}")
        service_quality = st.radio(
            "客服度",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: f"{x}分",
            horizontal=True,
            key=f"service__{key_prefix}",
        )
        total = score_total(service_quality)
        st.metric("总分（均值）", total)
        return {
            "service_quality": service_quality,
            "total": total,
        }

    left, right = st.columns(2)
    with left:
        score_a = render_model_score("音频 A 打分", f"{scene_id}__A")
    with right:
        score_b = render_model_score("音频 B 打分", f"{scene_id}__B")

    # 打分区底部导航（与顶部一致）
    bottom_prev, _, bottom_next = st.columns([1, 2, 1])
    with bottom_prev:
        if st.button("◀ 上一条", width="stretch", key=f"bottom_prev_{scene_id}"):
            score_a_nav = get_slot_score_from_state(scene_id, "A")
            score_b_nav = get_slot_score_from_state(scene_id, "B")
            stash_scene_score(scene_id, sample_id, dialogue_text, ab_assignment, score_a_nav, score_b_nav)
            go_prev_sample()
            st.rerun()
    with bottom_next:
        if st.button("下一条 ▶", width="stretch", key=f"bottom_next_{scene_id}"):
            score_a_nav = get_slot_score_from_state(scene_id, "A")
            score_b_nav = get_slot_score_from_state(scene_id, "B")
            stash_scene_score(scene_id, sample_id, dialogue_text, ab_assignment, score_a_nav, score_b_nav)
            go_next_sample(len(selected_scenes))
            st.rerun()

    # 也可不切换场景，手动暂存当前条
    if st.button("仅暂存当前评分", width="stretch"):
        stash_scene_score(scene_id, sample_id, dialogue_text, ab_assignment, score_a, score_b)
        st.success(f"已暂存：场景 {scene_id}")

    st.info(f"已暂存场景数：{len(st.session_state.staged_scores)} / {len(selected_scenes)}")

    # 最终一次提交全部暂存
    submit_col = st.columns([1, 2, 1])[1]
    with submit_col:
        if st.button("提交测评结果", width="stretch", type="primary"):
            if not st.session_state.staged_scores:
                st.warning("当前没有暂存评分。")
            else:
                results_all = load_results()

                # 防重复：同评测人+同场景+同模型组合先删除旧记录，再写入新记录
                staged_items = list(st.session_state.staged_scores.values())
                staged_index = {
                    (
                        evaluator_name,
                        item["scene_id"],
                        tuple(sorted(item["models"].keys())),
                    )
                    for item in staged_items
                }
                filtered = []
                for r in results_all:
                    k = (
                        r.get("evaluator"),
                        r.get("scene_id"),
                        tuple(sorted((r.get("models") or {}).keys()))
                        if isinstance(r.get("models"), dict)
                        else tuple(),
                    )
                    if k in staged_index:
                        continue
                    filtered.append(r)

                now_iso = datetime.now().isoformat()
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for item in sorted(staged_items, key=lambda x: x["sample_id"]):
                    filtered.append(
                        {
                            "evaluator": evaluator_name,
                            "sample_id": item["sample_id"],
                            "scene_id": item["scene_id"],
                            "dialogue": item["dialogue"],
                            "models": item["models"],
                            "timestamp": now_iso,
                            "date": now_str,
                        }
                    )
                write_results(filtered)
                st.success(f"已提交 {len(staged_items)} 条暂存评分。")
                st.session_state.staged_scores = {}


# -----------------------------
# Results Summary
# -----------------------------
elif page == "结果汇总":
    if evaluator_name != "任夏明":
        st.error("无权限访问该页面。")
        st.stop()
    st.header("评测结果汇总")
    results = load_results()
    if removed_legacy_count > 0:
        st.info(f"已自动清理旧版残留数据：{removed_legacy_count} 条。")

    if not results:
        st.info("暂无评测记录")
    else:
        df = pd.DataFrame(normalize_results(results))
        if df.empty:
            st.info("结果结构为空")
            st.stop()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总提交数", len(results))
        with col2:
            st.metric("评测人数", df["Evaluator"].dropna().nunique())
        with col3:
            avg_total = df["Total Score"].dropna().mean()
            st.metric("平均总分", f"{avg_total:.2f}" if pd.notna(avg_total) else "N/A")

        st.divider()
        st.subheader("明细")
        show_dataframe(df, width="stretch")

        st.subheader("按评测人/模型统计")
        grouped = (
            df.groupby(["Evaluator", "Model"], dropna=False)
            .agg(
                样本数=("Total Score", "count"),
                平均总分=("Total Score", "mean"),
                平均客服度=("客服度", "mean"),
            )
            .round(3)
        )
        show_dataframe(grouped, width="stretch")


# -----------------------------
# Data Export
# -----------------------------
elif page == "数据导出":
    if evaluator_name != "任夏明":
        st.error("无权限访问该页面。")
        st.stop()
    st.header("数据导出")
    results = load_results()

    st.download_button(
        label="下载 JSON",
        data=json.dumps(results, indent=2, ensure_ascii=False),
        file_name=f"mos_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )

    df = pd.DataFrame(normalize_results(results))
    csv_data = df.to_csv(index=False) if not df.empty else ""
    st.download_button(
        label="下载 CSV",
        data=csv_data,
        file_name=f"mos_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("原始结果预览")
    st.json(results if results else [])