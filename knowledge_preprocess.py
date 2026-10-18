# 集成说明：这是通用解析模板，支持类型多于网页上传白名单；网页实际允许的后缀
# 以knowledge_upload.ALLOWED_SUFFIXES为准，保存MinerU参数不会让本模块自动调用MinerU。
"""简易 RAG 资料预处理模块：把文件或已有文本整理为可入库的文本块。

完整流程：读取/解析 -> 保守清洗 -> 结构识别与分块 -> 来源记录 -> 导出。
这里负责 Agent 知识库的数据准备；embedding、向量库和问答由调用方负责。

三个统一入口：
    prepare(path, config)          读取文件，再执行清洗和分块。
    prepare_text(text, ...)        已经有文本/Markdown，直接执行清洗和分块。
    prepare_segments(pages, ...)   已有分页面的解析结果，保留页码后执行处理。

建议阅读顺序：PipelineConfig -> prepare -> _prepare -> clean_text -> split_text。
_blocks 是清洗与分块共用的结构识别函数；HTML/DOCX 解析可放到后面阅读。

基础功能只使用 Python 标准库，需要 Python >= 3.10；读取 PDF 可选安装 pypdf。
MinerU 不在这个文件里自动运行。若已经用 MinerU 解析好文档，可使用后两个入口。

注意三个边界：
1. 长度和偏移使用 Python 字符数，不是模型 token 数，也不是文件字节位置。
2. 字符偏移对应清洗后的 segment；不提供原 PDF 坐标或原文位置映射。
3. 这是基础模块，未实现 OCR、向量化、缓存或完整企业级数据安全能力。
"""
# 延后求值类型注解，便于在注解中引用类型，并保持运行时逻辑简洁。
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Iterable
import xml.etree.ElementTree as ET
import zipfile

# 算法版本也参与 chunk_id 计算；以后修改处理行为时应同步维护版本。
__version__ = "0.1.0"
# 对外公开的类型和函数。下划线开头的函数/类是内部实现，通常不用直接调用。
__all__ = ["PipelineConfig", "CleanRule", "Segment", "Chunk", "TextSpan",
           "CleaningResult", "PreparedResult", "PreprocessError", "DocumentReadError",
           "MissingDependencyError", "EmptyDocumentError", "prepare", "prepare_text",
           "prepare_segments", "clean_text", "split_text", "write_jsonl"]


# ---------------------------------------------------------------------------
# 1. 异常类型：让调用方能区分“处理失败”和“成功但有提示”。
#    可统一 except PreprocessError；配置错误则使用 Python 的 ValueError。
# ---------------------------------------------------------------------------

class PreprocessError(Exception):
    """资料预处理错误的共同父类，方便调用方统一捕获业务失败。"""


class DocumentReadError(PreprocessError):
    """读取或解析失败，例如文件不存在、格式不支持或内容损坏。"""


class MissingDependencyError(DocumentReadError):
    """缺少可选解析依赖，例如读取 PDF 时没有安装 pypdf。"""


class EmptyDocumentError(PreprocessError):
    """解析前后或清洗后没有可用文字，避免把空结果误报为成功。"""


# ---------------------------------------------------------------------------
# 2. 数据契约与配置：用明确的数据结构连接各处理步骤。
#    dataclass 自动生成构造函数；frozen=True 禁止常规字段重新赋值。
#    frozen 不会递归冻结字段内的字典，所以 Chunk.metadata 仍是可变字典。
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CleanRule:
    """一条开发者维护的正则清洗规则。

    例如 CleanRule("mask_id", r"CID-[0-9]{6}", "[CUSTOM_ID]")。
    普通规则只在非表格、非 fenced code 的单个结构块中执行。
    此类不提供正则超时或沙箱，不应直接接受不可信用户提供的任意规则。
    """
    name: str                        # 规则名：写入处理记录，必须非空且在配置中唯一。
    pattern: str                     # 正则表达式：描述要匹配的文字。
    replacement: str = ""            # 替换结果：空串表示删除，也可以填脱敏标签。
    priority: int = 100               # 数字越小越先执行；相同优先级保持原有顺序。
    flags: int = 0                    # 如 re.MULTILINE；默认不会开启跨行的 DOTALL。

    def __post_init__(self):
        """dataclass 构造完对象后立即检查规则，尽量提前暴露配置错误。"""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("规则 name 必须是非空字符串")
        if not isinstance(self.pattern, str) or not isinstance(self.replacement, str):
            raise ValueError("pattern 和 replacement 必须是字符串")
        if type(self.priority) is not int or not isinstance(self.flags, int):
            raise ValueError("规则 priority 和 flags 必须是整数")
        try:
            compiled = re.compile(self.pattern, self.flags)
            # 即使正文还没读取，也检查替换模板中的分组引用。
            # 例如 pattern 只有一个分组，replacement 却引用第二组，应提前报错。
            compiled.sub(self.replacement, "")
        except (re.error, ValueError, IndexError) as exc:
            raise ValueError(f"正则规则 {self.name!r} 无效: {exc}") from exc


@dataclass(frozen=True)
class PipelineConfig:
    """整个处理流程的配置；不同 Agent 项目可以复用代码、使用不同配置。

    表格/代码/长标题保持完整，因此 max_chunk_chars 是结构块的软上限。
    overlap_chars 只作用于需要滑窗切分的超长普通段落。
    """
    max_chunk_chars: int = 800        # 普通文本块的最大字符预算，不能当作 token 预算。
    overlap_chars: int = 80           # 超长普通段落相邻窗口重叠的字符数。
    split_by_heading: bool = True     # 遇到常见 Markdown 标题时开启新的章节分组。
    mask_pii: bool = False            # 显式开启手机号、身份证号、邮箱的启发式脱敏。
    remove_page_numbers: bool = True  # 仅删除占据整行的常见页码标记。
    encoding: str | None = None       # None 时依次尝试 UTF-8 和 GB18030。
    rules: tuple[CleanRule, ...] = ()  # 自定义规则集合；不会默认删除网址或版权文字。
    max_input_bytes: int = 20 * 1024 * 1024  # 20 MiB 输入限制；不能代替解析沙箱。

    def __post_init__(self):
        """校验尺寸、开关、编码和规则，拒绝容易造成错误分块的配置。"""
        # bool 是 int 的子类；使用 type(...) is int，避免误接受 True 作为长度。
        if type(self.max_chunk_chars) is not int or self.max_chunk_chars < 1:
            raise ValueError("max_chunk_chars 必须是正整数（字符数，不是 token 数）")
        if type(self.overlap_chars) is not int or not 0 <= self.overlap_chars < self.max_chunk_chars:
            raise ValueError("overlap_chars 必须满足 0 <= overlap_chars < max_chunk_chars")
        if type(self.max_input_bytes) is not int or self.max_input_bytes < 1:
            raise ValueError("max_input_bytes 必须是正整数")
        for name in ("split_by_heading", "mask_pii", "remove_page_numbers"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} 必须是布尔值")
        if self.encoding is not None:
            if not isinstance(self.encoding, str) or not self.encoding:
                raise ValueError("encoding 必须是编码名称或 None")
            try:
                "".encode(self.encoding)
            except LookupError as exc:
                raise ValueError(f"未知编码: {self.encoding}") from exc
        # 在 frozen 对象初始化阶段将传入规则固化成元组。
        # 这样调用方之后修改原来的列表，不会顺带改变这里的配置。
        object.__setattr__(self, "rules", tuple(self.rules))
        if any(not isinstance(rule, CleanRule) for rule in self.rules):
            raise ValueError("rules 中的每一项必须是 CleanRule")
        names = [rule.name for rule in self.rules]
        if len(names) != len(set(names)):
            raise ValueError("自定义规则名称不能重复")


@dataclass(frozen=True)
class Segment:
    """解析器输出的一页或一个逻辑片段，尚不是最终检索用的 Chunk。

    PDF 通常每页一个 Segment；TXT/HTML/DOCX 通常整份资料一个 Segment。
    只有解析器提供可靠页码时才填写 page，不能从分块序号猜测文档页码。
    """
    text: str                        # 当前页/段的完整文本；也可以包含 Markdown。
    page: int | None = None           # 从 1 开始的原文页码；不确定时保留 None。

    def __post_init__(self):
        """保证后续步骤获得字符串及有效页码。"""
        if not isinstance(self.text, str):
            raise ValueError("Segment.text 必须是字符串")
        if self.page is not None and (type(self.page) is not int or self.page < 1):
            raise ValueError("Segment.page 必须是从 1 开始的整数或 None")


@dataclass(frozen=True)
class CleaningResult:
    """一次 clean_text 调用的输出：正文、应用规则和脱敏匹配数量。"""
    text: str                        # 清洗后的完整文本。
    rules_applied: tuple[str, ...]    # 去重后的规则名，保留首次出现顺序。
    pii_count: int                   # 正则匹配并替换的数量，不代表真实隐私检出准确率。


@dataclass(frozen=True)
class TextSpan:
    """一个块在指定字符串中的范围，使用左闭右开区间 [start_char, end_char)。

    split_text 先返回范围，稍后 _prepare 再截取文本并补充来源和 ID。
    """
    start_char: int                   # 第一个字符的位置，包含该字符。
    end_char: int                     # 结束位置，不包含此位置上的字符。
    section_title: str | None         # 当前 Segment 内识别到的章节标题。
    oversized: bool = False           # 结构整体超过预算，向量化前需要额外检查。


@dataclass(frozen=True)
class Chunk:
    """交给 embedding/向量库的最终文本块，包含稳定 ID 和可追踪来源。"""
    chunk_id: str                     # 由来源、配置、版本、位置和文本生成的摘要。
    text: str                         # 完整正文，不截断为预览文本。
    metadata: dict = field(default_factory=dict)  # 每个实例创建独立字典，避免共享可变默认值。

    def to_dict(self) -> dict:
        """递归转换为普通字典，便于 JSONL 导出或向量库适配。"""
        return asdict(self)


@dataclass
class PreparedResult:
    """整份资料的统一处理结果；三个入口都返回这个类型。"""
    source: str                       # 文件绝对路径或调用方提供的稳定来源标识。
    source_hash: str                  # 输入内容摘要；具体计算对象由入口决定。
    cleaned_segments: list[Segment]   # 完整清洗正文，供检查内容与核对字符偏移。
    chunks: list[Chunk]               # 最终可入库的块列表。
    report: dict                      # 长度、规则、PII 数量、耗时和提示信息。

    def to_dict(self) -> dict:
        """转换整个结果，保留全部正文；大文档序列化后也可能很大。"""
        return asdict(self)


@dataclass(frozen=True)
class _Block:
    """内部结构块：用于区分普通段落、标题、表格和代码，不直接作为入库结果。"""
    start: int
    end: int
    kind: str
    title: str | None = None


# ---------------------------------------------------------------------------
# 3. 基础结构识别：只识别常见 Markdown，不是完整的 Markdown 解析器。
# ---------------------------------------------------------------------------

# 识别 # 到 ###### 标题，允许开头最多三个空格。
_HEADING = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)\s*$")
# 代码围栏使用至少三个反引号或波浪号；后面允许语言标记，如 ```python。
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
# 管道表格的分隔行，如 | --- | :---: |；此处也兼容单列表格。
# 这里不会把 Markdown 中嵌入的 HTML <table> 当作受保护表格。
_TABLE_SEPARATOR = re.compile(r"^\s*(?=.*\|)\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")
# 调用时使用 fullmatch，确保只删除整行页码，不删除“请参见第2页”这样的正文。
_PAGE = re.compile(r"(?:第\s*[0-9]+\s*页(?:\s*[/／]\s*(?:共\s*)?[0-9]+\s*页?)?|"
                   r"页码[：:]\s*[0-9]+\s*[/／]\s*[0-9]+|"
                   r"Page\s+[0-9]+(?:\s+of\s+[0-9]+)?|---\s*Page\s+[0-9]+\s*---)", re.I)


def _config(config: PipelineConfig | None) -> PipelineConfig:
    """统一配置入口：未传入时创建默认配置，传错类型则明确报错。"""
    if config is None:
        return PipelineConfig()
    if not isinstance(config, PipelineConfig):
        raise ValueError("config 必须是 PipelineConfig 或 None")
    return config


def _blocks(text: str) -> list[_Block]:
    """按行识别结构块，并记录它们在传入 text 中的精确字符范围。

    返回的块不包含块外空白行；代码内部的空白行和缩进仍完整保留。
    清洗阶段利用 kind 避开表格/代码，分块阶段利用 kind 保持结构整体。
    """
    # 保留行结束符才能计算原字符串的偏移；不能先 split 后丢失换行长度。
    lines = text.splitlines(keepends=True)
    # positions[i] 是第 i 行开头的字符位置；最后一项是整个文本的结束位置。
    positions = [0]
    for line in lines:
        positions.append(positions[-1] + len(line))

    def table_at(index):
        """同时检查表头行和下一行分隔线，避免把任意含 | 的正文当成表格。"""
        return (index + 1 < len(lines) and "|" in lines[index]
                and bool(_TABLE_SEPARATOR.fullmatch(lines[index + 1].rstrip("\r\n"))))

    result = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\r\n")
        if not line.strip():
            i += 1
            continue
        start = i
        heading = _HEADING.match(line)
        fence = _FENCE.match(line)
        kind, title = "prose", None
        if fence:
            # 从开围栏一直扫描到匹配的闭围栏；没有闭围栏时保护剩余整个片段。
            kind = "code"
            marker = fence.group(1)
            # 结束围栏必须使用相同字符，而且长度不能短于开围栏。
            closer = re.compile(r"^ {0,3}" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}\s*$")
            i += 1
            while i < len(lines):
                closes = bool(closer.fullmatch(lines[i].rstrip("\r\n")))
                i += 1
                if closes:
                    break
        elif table_at(i):
            # 把表头、分隔线和后续连续表格行当成一个整体。
            kind = "table"
            i += 2
            while i < len(lines) and lines[i].strip() and "|" in lines[i]:
                i += 1
        elif heading:
            kind = "heading"
            # 去掉可选的尾部标题井号，如“## 费用 ##”记录为“费用”。
            title = re.sub(r"\s+#+\s*$", "", heading.group(2)).strip()
            i += 1
        else:
            # 连续正文行合成一个段落，遇到空行或新结构块时停止。
            i += 1
            while i < len(lines) and lines[i].strip():
                current = lines[i].rstrip("\r\n")
                if _HEADING.match(current) or _FENCE.match(current) or table_at(i):
                    break
                i += 1
        end = positions[i]
        # 只排除块末尾的行结束符，不删除代码缩进或其他有效字符。
        while end > positions[start] and text[end - 1] in "\r\n":
            end -= 1
        result.append(_Block(positions[start], end, kind, title))
    return result


# ---------------------------------------------------------------------------
# 4. 清洗：先按结构规整正文，再按需执行 PII（个人信息）脱敏。
# ---------------------------------------------------------------------------

def clean_text(text: str, config: PipelineConfig | None = None) -> CleaningResult:
    """保守清洗一段文本；输入字符串不会被原地修改。

    参数：text 为原文；config 控制页码移除、自定义规则和脱敏开关。
    返回：CleaningResult，含清洗全文、应用规则及 PII 替换数量。

    普通正则规则跳过表格和 fenced code；显式开启的 PII 脱敏处理所有正文。
    空文本可以返回空的 CleaningResult；统一 prepare 入口会进一步拒绝空文档。
    脱敏属于启发式检测，不保证真实隐私信息无漏检或无误检。
    """
    cfg = _config(config)
    if not isinstance(text, str):
        raise ValueError("text 必须是字符串")
    # 统一 Windows/旧式换行并去掉 NUL，后续算法只需处理 \n。
    # 这些基础替换也作用于代码块，所以本模块不承诺原始文件逐字节不变。
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    applied = []
    parts = []
    # 一次调用中只编译一次规则；随后按优先级依次处理各个普通结构块。
    compiled = [(rule, re.compile(rule.pattern, rule.flags))
                for rule in sorted(cfg.rules, key=lambda item: item.priority)]
    for block in _blocks(normalized):
        part = normalized[block.start:block.end]
        if block.kind not in ("code", "table"):
            # 行末空白可以规整，但不能随便剥掉代码内部缩进或表格中的排版。
            lines = []
            for line in part.split("\n"):
                if cfg.remove_page_numbers and _PAGE.fullmatch(line.strip()):
                    applied.append("remove_page_numbers")
                else:
                    lines.append(line.rstrip())
            part = "\n".join(lines)
            for rule, regex in compiled:
                try:
                    replacement = regex.sub(rule.replacement, part)
                except (re.error, IndexError) as exc:
                    raise ValueError(f"规则 {rule.name!r} 的 replacement 无效: {exc}") from exc
                if replacement != part:
                    # 只有实际改变了正文，才记录这条用户规则。
                    applied.append(rule.name)
                part = replacement
            part = part.strip("\n")
        if part.strip():
            parts.append(part)
    cleaned = "\n\n".join(parts)
    # normalize_text 是非 PII 清洗前后的总体变化标记，可能包含自定义规则造成的变化。
    if cleaned != text:
        applied.insert(0, "normalize_text")
    pii_count = 0
    if cfg.mask_pii:
        # 先替换长身份证号，再处理邮箱、手机号，减少号码模式之间的重复计数。
        # 数字边界使用 [0-9]，允许匹配紧挨中文标签的“电话138...”形式。
        # 此处不校验身份证校验位，也不覆盖姓名、地址、银行卡等其他类别。
        patterns = [
            (r"(?<![0-9])[0-9]{17}[0-9Xx](?![0-9A-Za-z])", "[ID_CARD]"),
            (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]"),
            (r"(?<![0-9])1[3-9][0-9]{9}(?![0-9])", "[PHONE]"),
        ]
        for pattern, replacement in patterns:
            # subn 同时返回替换后的字符串和匹配数量，用于报告 pii_count。
            cleaned, count = re.subn(pattern, replacement, cleaned)
            if count:
                pii_count += count
                applied.append("mask_" + replacement[1:-1].lower())
    # dict 保留插入顺序：去重后仍能看到规则首次应用的顺序。
    return CleaningResult(cleaned, tuple(dict.fromkeys(applied)), pii_count)


# ---------------------------------------------------------------------------
# 5. 分块：完整段落优先，超长普通段落才使用有重叠的滑窗。
# ---------------------------------------------------------------------------

def _prose_spans(text: str, start: int, end: int, cfg: PipelineConfig, title: str | None):
    """生成超长普通段落的 TextSpan；start/end 对应整份传入 text 的偏移。"""
    while start < end:
        # 先确定一个不超过预算的硬边界，再尝试向前移动到更自然的位置。
        cut = min(start + cfg.max_chunk_chars, end)
        if cut < end:
            # 只在窗口后半段寻找边界，避免一遇到短句就生成大量非常短的块。
            lower = start + cfg.max_chunk_chars // 2
            sentence_cuts = [start + match.end() for match in re.finditer(r"[。！？.!?]\s*", text[start:cut])
                             if start + match.end() > lower]
            whitespace_cuts = [start + match.end() for match in re.finditer(r"\s+", text[start:cut])
                               if start + match.end() > lower]
            # 优先句末，其次空白；两者都没有时维持按字符切分。
            # 这是简易标点启发式，不是语言模型驱动的语义分块。
            candidates = sentence_cuts or whitespace_cuts
            if candidates:
                cut = candidates[-1]
        yield TextSpan(start, cut, title)
        if cut == end:
            break
        # 回退 overlap_chars 保留上下文，但窗口起点必须至少前进一个字符。
        # 否则在很短句子或很大重叠量下，可能反复处理同一窗口而无法结束。
        start = max(start + 1, cut - cfg.overlap_chars)


def split_text(text: str, config: PipelineConfig | None = None) -> list[TextSpan]:
    """按标题/段落组织范围；此函数本身不清洗、不创建来源 ID。

    参数：text 是需要分块的精确字符串，通常来自 clean_text。
    返回：TextSpan 列表，正文可由 text[span.start_char:span.end_char] 获取。

    短段落尽量合并；超长普通段落滑窗切分；表格、代码和长标题完整保留。
    只有超长普通段落窗口间有重叠，普通块、不同章节和不同 Segment 不加重叠。
    未传入有用内容时返回空列表；统一入口负责判断空结果是否应报错。
    """
    cfg = _config(config)
    if not isinstance(text, str):
        raise ValueError("text 必须是字符串")
    spans = []
    pending: list[_Block] = []         # 尚未提交的短结构块，尝试合并到同一 Chunk。
    title = None

    def flush():
        """把当前待合并块变成一个范围，然后清空缓冲；无需复制正文。"""
        if pending:
            spans.append(TextSpan(pending[0].start, pending[-1].end, title))
            pending.clear()

    for block in _blocks(text):
        if block.kind == "heading" and cfg.split_by_heading:
            # 先提交上一章节，再更新 title，保证旧块不会被误标成新章节。
            flush()
            title = block.title
        size = block.end - block.start
        if size > cfg.max_chunk_chars:
            flush()
            if block.kind == "prose":
                spans.extend(_prose_spans(text, block.start, block.end, cfg, title))
            else:
                # 不为了满足字符预算切坏表格/代码/标题；交由调用方处理 oversized。
                spans.append(TextSpan(block.start, block.end, title, oversized=True))
        else:
            # 合并后的长度包含块之间的空白行，避免忽略分隔符而超预算。
            if pending and block.end - pending[0].start > cfg.max_chunk_chars:
                flush()
            pending.append(block)
    flush()
    return spans


# ---------------------------------------------------------------------------
# 6. 统一编排与来源记录：三个入口最终都汇入 _prepare。
# ---------------------------------------------------------------------------

def _digest(data: bytes) -> str:
    """计算内容 SHA-256；用于标识内容，不代替访问控制或防篡改审计。"""
    return hashlib.sha256(data).hexdigest()


def _json(data) -> str:
    """稳定序列化：字典键排序、固定分隔符，避免键顺序改变哈希结果。"""
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def prepare_segments(segments: Iterable[Segment], *, source: str,
                     config: PipelineConfig | None = None,
                     source_hash: str | None = None) -> PreparedResult:
    """接收外部解析器产生的 Segment，例如 MinerU/OCR 提供的分页面文本。

    参数：segments 为可迭代的 Segment；source 为稳定来源名称或路径。
    可选 source_hash 应为原文件的完整 SHA-256；不传则摘要覆盖文字/页码记录。
    返回：PreparedResult，包括清洗全文、块、来源元数据和报告。

    这里不调用 MinerU，也不自动识别其输出格式；由调用方转换成 Segment。
    保留可靠页码时应设置 Segment.page，而不是从 chunk_index 推测页码。
    """
    cfg = _config(config)
    # 提前物化可迭代对象，便于重复校验、计算摘要和执行后续处理。
    records = list(segments)
    if any(not isinstance(segment, Segment) for segment in records):
        raise ValueError("segments 必须包含 Segment 对象")
    if source_hash is None:
        source_hash = _digest(_json([asdict(segment) for segment in records]).encode("utf-8"))
    return _prepare(records, source, source_hash, cfg, [], "external_segments")


def prepare_text(text: str, *, source: str = "<memory>",
                 config: PipelineConfig | None = None) -> PreparedResult:
    """已经有文本时使用的入口，跳过文件读取阶段。

    例如 MinerU 返回整份 Markdown 后，调用 prepare_text(markdown, source=文件名)。
    source_hash 在此入口覆盖原 text 的 UTF-8 字节，不能当作原 PDF 文件摘要。
    整体文本封装成一个无页码的 Segment；返回类型与 prepare 完全相同。
    """
    cfg = _config(config)
    segment = Segment(text)
    if len(text.encode("utf-8")) > cfg.max_input_bytes:
        raise DocumentReadError("文本超过 max_input_bytes 限制")
    return _prepare([segment], source, _digest(text.encode("utf-8")), cfg, [], "text")


def _prepare(segments, source, source_hash, cfg, warnings, parser):
    """核心编排：校验输入 -> 逐段清洗 -> 分块 -> 补元数据 -> 汇总报告。

    segments 是已经解析的内容；parser 仅作为报告中的解析器名称。
    warnings 是当前调用专用的提示列表，此函数会追加空页和超长结构块提示。
    这里只计清洗/分块耗时；文件入口随后会改成包含文件读取/解析的总耗时。
    """
    started = time.perf_counter()
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source 必须是非空字符串")
    if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise ValueError("source_hash 必须是完整的小写 SHA-256 十六进制摘要")
    if not segments or not any(segment.text.strip() for segment in segments):
        raise EmptyDocumentError("没有可用文字；扫描 PDF 需要先经过 OCR/MinerU 解析")
    if sum(len(segment.text.encode("utf-8")) for segment in segments) > cfg.max_input_bytes:
        raise DocumentReadError("解析后的文字超过 max_input_bytes 限制")
    # 同样的输入用了不同规则或分块设置，配置摘要也不同，便于追踪处理差异。
    config_hash = _digest(_json(asdict(cfg)).encode("utf-8"))
    # 来源也参与文档标识：内容相同但来自不同资料，不会被当作同一文档。
    doc_id = _digest(_json([source, source_hash]).encode("utf-8"))
    cleaned_segments, chunks, rules = [], [], []
    pii_count = 0
    for index, segment in enumerate(segments):
        cleaned = clean_text(segment.text, cfg)
        # 空页也保留其 Segment，使后续索引不会因删除空页而错位。
        cleaned_segments.append(Segment(cleaned.text, segment.page))
        pii_count += cleaned.pii_count
        rules.extend(cleaned.rules_applied)
        if not cleaned.text.strip():
            warnings.append(f"第 {segment.page or index + 1} 页/段没有可用文字，未生成块")
            continue
        for span in split_text(cleaned.text, cfg):
            chunk_text = cleaned.text[span.start_char:span.end_char]
            # ID 不包含当前时间，重复处理相同来源/内容/配置/版本会得到相同 ID。
            # 这是供调用方去重/upsert 的标识，本模块不会自动执行缓存或入库。
            identity = [__version__, doc_id, config_hash, index,
                        span.start_char, span.end_char, chunk_text]
            chunk_id = _digest(_json(identity).encode("utf-8"))
            # page 从 1 开始；chunk_index 和 segment_index 从 0 开始。
            # start_char/end_char 永远指向对应 cleaned_segments[index].text。
            meta = {"source": source, "source_hash": source_hash, "doc_id": doc_id,
                    "pipeline_version": __version__, "config_hash": config_hash,
                    "chunk_index": len(chunks), "segment_index": index,
                    "page": segment.page, "section_title": span.section_title,
                    "start_char": span.start_char, "end_char": span.end_char,
                    "offset_basis": "cleaned_segments[segment_index].text",
                    "oversized": span.oversized}
            chunks.append(Chunk(chunk_id, chunk_text, meta))
            if span.oversized:
                warnings.append(f"块 {len(chunks) - 1} 保持表格/代码/标题完整，"
                                f"长度 {len(chunk_text)} 超过 {cfg.max_chunk_chars} 字符")
    if not chunks:
        raise EmptyDocumentError("清洗后没有可用文字，请检查清洗规则")
    original_chars = sum(len(segment.text) for segment in segments)
    cleaned_chars = sum(len(segment.text) for segment in cleaned_segments)
    # 原始长度是“解析后的文字长度”，不是原文件体积。
    # retention_ratio 是字符保留比例，不是 token 成本或清洗准确率。
    # success 表示产出了可用块；warnings 仍可能说明空页、结构超长等局部问题。
    report = {"status": "success", "source": source, "parser": parser,
              "pipeline_version": __version__, "source_hash": source_hash,
              "config_hash": config_hash, "segment_count": len(segments),
              "chunk_count": len(chunks), "original_chars": original_chars,
              "cleaned_chars": cleaned_chars,
              "character_retention_ratio": round(cleaned_chars / max(original_chars, 1), 4),
              "pii_detected_count": pii_count, "rules_applied": list(dict.fromkeys(rules)),
              "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
              "warnings": list(dict.fromkeys(warnings))}
    return PreparedResult(source, source_hash, cleaned_segments, chunks, report)


# ============================================================================
# 7. 基础文件解析：把不同格式统一转换为文字 Segment
# ============================================================================

def _decode(data: bytes, encoding: str | None) -> str:
    """将文件字节解码为文字；指定编码优先，否则依次尝试两种常见编码。"""
    if encoding:
        try:
            return data.decode(encoding)
        except UnicodeError as exc:
            raise DocumentReadError(f"文件不是有效的 {encoding} 编码") from exc
    # utf-8-sig 同时兼容普通 UTF-8 和带 BOM 的 UTF-8；GB18030 兼容常见中文编码。
    # 这是按顺序尝试，不是精确的编码识别；遇到乱码时应由调用者指定 encoding。
    for candidate in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(candidate)
        except UnicodeError:
            pass
    raise DocumentReadError("无法按 UTF-8 或 GB18030 解码；请显式指定 encoding")


def _markdown_table(rows: list[list[str]]) -> str:
    """把二维单元格列表转为管道式 Markdown 表格，供后续结构分块识别。"""
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    if width == 1:
        # 当前输出约定：单列表格补一列空值，使表格表示保持一致。
        width = 2
    output = []
    for index, row in enumerate(rows):
        # 转义单元格中的竖线，避免被误当成列边界；换行转为 <br> 保持一行一条记录。
        cells = [cell.replace("|", r"\|").replace("\n", "<br>") for cell in row]
        cells.extend([""] * (width - len(cells)))
        output.append("| " + " | ".join(cells) + " |")
        if index == 0:
            # Markdown 要求第一行后有分隔行；这里只把第一行用作表头，未判断其业务含义。
            output.append("| " + " | ".join(["---"] * width) + " |")
    return "\n".join(output)


class _BasicHTML(HTMLParser):
    """用标准库读取基础 HTML 正文、标题、表格和代码，输出近似 Markdown。

    HTMLParser 会按顺序回调下面的 handle_* 方法；实例字段用于记录当前所在结构。
    这里只读取标签与文字，不执行 JavaScript，也不恢复浏览器布局。
    """
    # 空元素没有正常的结束标签，不能把它们压入“忽略区域”的标签栈。
    _VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
             "link", "meta", "param", "source", "track", "wbr"}
    _PARAGRAPH = {"p", "div", "section", "article", "li", "ul", "ol", "blockquote"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []       # 已转换好的正文片段，最后按顺序拼接。
        self.ignored = []     # script/style/nav 等跳过区域内的嵌套标签栈。
        self.rows = None      # None 表示不在表格中；[] 表示已进入尚无行的表格。
        self.row = None       # 当前行的单元格列表。
        self.cell = None      # 当前单元格收集到的文字片段。
        self.pre = None       # 当前 <pre> 代码片段；保留代码空格与换行。
        self.caption = None   # 当前表格标题的文字片段。

    def _finish_cell(self):
        """将已收集的单元格加入当前行；也兼容省略部分表格标签的 HTML。"""
        if self.cell is not None:
            if self.row is None:
                self.row = []
            self.row.append("".join(self.cell).strip())
            self.cell = None

    def _finish_row(self):
        """先收尾当前单元格，再将当前行加入表格。"""
        self._finish_cell()
        if self.row is not None:
            self.rows.append(self.row)
            self.row = None

    def handle_starttag(self, tag, attrs):
        """处理开始标签：优先判断忽略区域、代码和表格，再处理普通正文。"""
        attrs = dict(attrs)
        if self.ignored:
            # 进入跳过区域后，其内部文字和标签都不应成为知识库内容。
            if tag not in self._VOID:
                self.ignored.append(tag)
            return
        if tag in {"script", "style", "nav"}:
            self.ignored.append(tag)
            return
        if self.pre is not None:
            # <pre> 内不转换普通标签，仅将 <br> 记作代码换行。
            if tag == "br":
                self.pre.append("\n")
            return
        if tag == "table":
            if self.rows is not None:
                raise DocumentReadError("简易 HTML 解析不支持嵌套表格")
            self.parts.append("\n\n")
            self.rows = []
        elif self.rows is not None:
            if tag == "caption":
                self.caption = []
            elif tag == "tr":
                # 新行开始时先收尾上一行，可兼容没有显式 </tr> 的写法。
                self._finish_row()
                self.row = []
            elif tag in {"td", "th"}:
                # 同样在新单元格开始时收尾上一个单元格。
                self._finish_cell()
                self.cell = []
            elif tag == "br" and self.cell is not None:
                self.cell.append("\n")
            elif tag in self._PARAGRAPH and self.cell:
                if not self.cell[-1].endswith("\n"):
                    self.cell.append("\n")
        elif tag == "pre":
            self.pre = []
        elif re.fullmatch(r"h[1-6]", tag):
            # h1～h6 映射为 Markdown 标题，使后面的分块器能识别章节。
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in self._PARAGRAPH:
            self.parts.append("\n\n")
        elif tag in {"br", "hr"}:
            self.parts.append("\n")
        elif tag == "img" and attrs.get("alt"):
            # 只保留已有的替代文字；不读取图片，也不做图片识别。
            self.parts.append("[图片：" + attrs["alt"] + "]")

    def handle_endtag(self, tag):
        """处理结束标签，并在代码或表格结束时生成对应 Markdown。"""
        if self.ignored:
            if tag in self.ignored:
                # 找到最近的同名开始标签，移除它及其内部尚未闭合的层次。
                del self.ignored[len(self.ignored) - 1 - self.ignored[::-1].index(tag):]
            return
        if self.pre is not None:
            if tag == "pre":
                content = "".join(self.pre)
                longest = max((len(m.group()) for m in re.finditer(r"`+", content)), default=0)
                # 围栏比代码内最长的连续反引号更长，避免代码内容意外关闭围栏。
                fence = "`" * max(3, longest + 1)
                self.parts.append("\n\n" + fence + "\n" + content.rstrip("\n") + "\n" + fence + "\n\n")
                self.pre = None
            return
        if self.rows is not None:
            if tag == "caption" and self.caption is not None:
                self.parts.append("".join(self.caption).strip() + "\n\n")
                self.caption = None
            elif tag in {"td", "th"} and self.cell is not None:
                self._finish_cell()
            elif tag == "tr" and self.row is not None:
                self._finish_row()
            elif tag == "table":
                # 结束表格时收尾尚未闭合的行/单元格，然后重置状态供下一张表使用。
                self._finish_row()
                if self.caption is not None:
                    self.parts.append("".join(self.caption).strip() + "\n\n")
                    self.caption = None
                self.parts.append(_markdown_table(self.rows) + "\n\n")
                self.rows = self.row = self.cell = None
            elif tag in self._PARAGRAPH and self.cell:
                if not self.cell[-1].endswith("\n"):
                    self.cell.append("\n")
            return
        if tag in self._PARAGRAPH or re.fullmatch(r"h[1-6]", tag):
            self.parts.append("\n\n")

    def handle_startendtag(self, tag, attrs):
        """处理 <tag /> 写法；空元素只需执行开始标签逻辑。"""
        self.handle_starttag(tag, attrs)
        if tag not in self._VOID:
            self.handle_endtag(tag)

    def handle_data(self, data):
        """按当前状态分发文字：代码原样收集，普通 HTML 空白折叠为空格。"""
        if self.ignored:
            return
        if self.pre is not None:
            self.pre.append(data)
        elif self.caption is not None:
            self.caption.append(re.sub(r"\s+", " ", data))
        elif self.cell is not None:
            self.cell.append(re.sub(r"\s+", " ", data))
        elif self.rows is None:
            self.parts.append(re.sub(r"\s+", " ", data))

    def text(self):
        """返回解析结果；表格或代码未闭合时明确报错，避免悄悄丢失内容。"""
        if self.rows is not None or self.pre is not None:
            raise DocumentReadError("HTML 的 table/pre 标签未闭合；请修正文档或使用高级解析器")
        # 正文规范化交给 clean_text；这里不能统一 strip，否则可能破坏代码缩进。
        return "".join(self.parts)


# DOCX 中 Word 正文 XML 的命名空间；ElementTree 使用“{命名空间}标签名”查找节点。
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx_text(data: bytes, limit: int) -> str:
    """读取 DOCX 主正文 XML，按原有顺序提取段落、基础标题和表格。

    DOCX 本质上是 ZIP 包。本函数只解析 word/document.xml，不能还原全部 Word 排版。
    limit 同时约束正文 XML 的解压后大小，不仅检查外层 DOCX 文件大小。
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            info = archive.getinfo("word/document.xml")
            if info.file_size > limit:
                raise DocumentReadError("DOCX 正文 XML 超过 max_input_bytes 限制")
            root = ET.fromstring(archive.read(info))
        body = root.find(_W + "body")
        if body is None:
            raise DocumentReadError("DOCX 缺少正文 body")

        def paragraph(node):
            """收集段落中的文字、制表符与换行；识别 Heading1～Heading6 样式。"""
            parts = []
            for child in node.iter():
                if child.tag == _W + "t":
                    parts.append(child.text or "")
                elif child.tag == _W + "tab":
                    parts.append("\t")
                elif child.tag in {_W + "br", _W + "cr"}:
                    parts.append("\n")
            content = "".join(parts)
            style = node.find("./" + _W + "pPr/" + _W + "pStyle")
            if style is not None:
                match = re.fullmatch(r"Heading([1-6])", style.get(_W + "val", ""), re.I)
                if match:
                    content = "#" * int(match.group(1)) + " " + content
            return content

        def children(node, accepted):
            """按文档顺序遍历目标节点，并进入支持的内容控件等包装节点。"""
            for child in node:
                if child.tag in accepted:
                    yield child
                elif child.tag == _W + "sdt":
                    content = child.find(_W + "sdtContent")
                    if content is not None:
                        yield from children(content, accepted)
                elif child.tag in {_W + "customXml", _W + "ins"}:
                    yield from children(child, accepted)
                elif child.tag == _W + "altChunk":
                    # 外部嵌入内容不在本解析器能力内，不能假装已完整读取。
                    raise DocumentReadError("DOCX 含外部 altChunk 内容，请使用高级解析器")

        parts = []
        for node in children(body, {_W + "p", _W + "tbl"}):
            # 在同一次遍历中处理段落和表格，保留它们在正文中的先后顺序。
            if node.tag == _W + "p":
                parts.append(paragraph(node))
            elif node.tag == _W + "tbl":
                rows = []
                for row in children(node, {_W + "tr"}):
                    cells = []
                    for cell in children(row, {_W + "tc"}):
                        contents = list(children(cell, {_W + "p", _W + "tbl"}))
                        if any(item.tag == _W + "tbl" for item in contents):
                            # 简单二维表无法表达嵌套表格，因此明确拒绝而不静默展平。
                            raise DocumentReadError("简易 DOCX 解析不支持嵌套表格，请使用高级解析器")
                        cells.append("\n".join(paragraph(p) for p in contents))
                    rows.append(cells)
                parts.append(_markdown_table(rows))
        return "\n\n".join(parts)
    except (zipfile.BadZipFile, KeyError, ET.ParseError, RuntimeError) as exc:
        raise DocumentReadError(f"DOCX 解析失败: {exc}") from exc


def prepare(path: str | Path, config: PipelineConfig | None = None) -> PreparedResult:
    """文件入口：读取文档，再完成清洗、分块和来源记录。

    path：本地文件路径，支持 TXT、Markdown、HTML、DOCX 和文字型 PDF。
    config：可选配置；不传时使用 PipelineConfig 的默认值。
    返回：PreparedResult，包含清洗后的片段、分块列表和处理报告。

    PDF 使用可选依赖 pypdf，仅提取文字；扫描件 OCR、复杂表格/图片需外部解析器。
    DOCX 只读取主正文的基础结构，无法完整还原版面。这里不会自动调用 MinerU。
    如使用 MinerU，先在外部解析，再将结果交给 prepare_text 或 prepare_segments。
    """
    cfg = _config(config)
    started = time.perf_counter()
    file = Path(path).resolve()
    # 统一绝对路径，便于追踪来源；通过后缀选择解析器，不是对文件内容进行格式鉴定。
    suffix = file.suffix.lower()
    if suffix not in {".txt", ".md", ".markdown", ".html", ".htm", ".docx", ".pdf"}:
        raise DocumentReadError(f"不支持的文件类型: {suffix or '(没有扩展名)'}")
    try:
        if not file.is_file():
            raise DocumentReadError(f"文件不存在或不是普通文件: {file}")
        if file.stat().st_size > cfg.max_input_bytes:
            raise DocumentReadError("文件超过 max_input_bytes 限制")
        with file.open("rb") as stream:
            # 即使文件在 stat 后变大，也只读取“上限 + 1”字节并再次检查。
            data = stream.read(cfg.max_input_bytes + 1)
        if len(data) > cfg.max_input_bytes:
            raise DocumentReadError("文件超过 max_input_bytes 限制")
    except OSError as exc:
        raise DocumentReadError(f"读取文件失败: {exc}") from exc
    warnings = []
    if suffix == ".pdf":
        try:
            # 延迟导入：只处理 TXT 等格式时，不要求安装 PDF 依赖。
            from pypdf import PdfReader
        except ImportError as exc:
            raise MissingDependencyError('PDF 需要可选依赖：python -m pip install "pypdf>=5"') from exc
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise DocumentReadError("此简易版本不支持加密 PDF，请先解密")
            segments = [Segment(page.extract_text() or "", index)
                        for index, page in enumerate(reader.pages, 1)]
            # 每页单独形成 Segment，页码从 1 开始；空页由公共流水线记录提示。
        except DocumentReadError:
            raise
        except Exception as exc:
            raise DocumentReadError(f"PDF 解析失败: {exc}") from exc
        warnings.append("PDF 仅提取文字，未做 OCR、表格结构恢复或图片提取")
        parser = "pypdf"
    elif suffix == ".docx":
        segments = [Segment(_docx_text(data, cfg.max_input_bytes))]
        warnings.append("DOCX 仅提取正文段落和基础表格；不含图片、页眉页脚、批注或精确页码")
        parser = "docx_xml"
    elif suffix in {".html", ".htm"}:
        html = _BasicHTML()
        html.feed(_decode(data, cfg.encoding))
        html.close()
        segments = [Segment(html.text())]
        warnings.append("HTML 为基础正文/表格提取；不执行 JavaScript，不完整保留复杂布局和合并单元格")
        parser = "html_stdlib"
    else:
        # TXT 和 Markdown 不做额外结构转换，交给后续清洗与结构识别。
        segments = [Segment(_decode(data, cfg.encoding))]
        parser = "text"
    result = _prepare(segments, str(file), _digest(data), cfg, warnings, parser)
    # 文件入口的总耗时包含文件读取和解析；source_hash 来自原文件字节。
    result.report["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result


# ============================================================================
# 8. 文件导出与命令行入口
# ============================================================================

def _atomic_write(path: Path, content: str):
    """先在目标目录写临时文件，再一次性替换目标文件，避免留下半写入结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=".rag-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        # with 已关闭临时文件，Windows 下才能安全替换；同目录避免跨文件系统移动。
        os.replace(temporary, path)
    finally:
        # 替换失败时清理临时文件；替换成功后原临时路径已经不存在。
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_jsonl(result: PreparedResult, path: str | Path) -> Path:
    """导出 UTF-8 JSONL：每行一个 Chunk 字典，并返回输出的绝对路径。

    已有输出会被替换；拒绝将结果写到记录中的输入路径，以免覆盖原文档。
    JSONL 便于逐块读取，后续可将 text 正文向量化，并保存对应的来源元数据。
    """
    destination = Path(path).resolve()
    if os.path.normcase(str(destination)) == os.path.normcase(result.source):
        raise ValueError("输出路径不能覆盖输入文件")
    _atomic_write(destination, "".join(_json(chunk.to_dict()) + "\n" for chunk in result.chunks))
    return destination


def _main() -> int:
    """把命令行参数转换为配置，再执行文件处理与导出；成功返回 0，失败返回 1。

    示例：python knowledge_preprocess.py knowledge.md -o chunks.jsonl --mask-pii
    使用 import 调用本模块时不会运行此函数，也不会自动创建输出文件。
    """
    parser = argparse.ArgumentParser(description="本地文档清洗与分块（长度按字符计算）")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--overlap-chars", type=int, default=80)
    parser.add_argument("--mask-pii", action="store_true")
    parser.add_argument("--encoding")
    args = parser.parse_args()
    try:
        cfg = PipelineConfig(max_chunk_chars=args.max_chars, overlap_chars=args.overlap_chars,
                             mask_pii=args.mask_pii, encoding=args.encoding)
        result = prepare(args.input, cfg)
        output = args.output or args.input.with_name(args.input.stem + ".chunks.jsonl")
        # 未指定 -o 时，输出放在输入文件旁；每个块独占 JSONL 中的一行。
        output = write_jsonl(result, output)
        report_path = output.with_suffix(".report.json")
        # 报告单独保存便于检查。本文件和 JSONL 各自原子写入，不构成双文件事务。
        _atomic_write(report_path, json.dumps(result.report, ensure_ascii=False, indent=2) + "\n")
        print(f"已生成 {len(result.chunks)} 个块：{output}")
        print(f"报告：{report_path}")
        for warning in result.report["warnings"]:
            print("提示：" + warning)
        return 0
    except (PreprocessError, ValueError, OSError) as exc:
        import sys
        print(f"处理失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # 仅直接运行脚本时进入 CLI；强制 UTF-8 输出，减少 Windows 终端中文编码问题。
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(_main())
