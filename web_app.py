"""本地网页入口：python web_app.py，浏览器访问 http://127.0.0.1:8765。"""
import asyncio
import hashlib
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import secrets
import sys

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from knowledge_upload import default_library, MAX_UPLOAD_BYTES, UploadError
from knowledge_bases import KnowledgeBases, DEFAULT_BASE
from service_settings import ServiceSettings

from agent_step1 import ALLOWED_TOOLS, initial_messages, run_turn
from rag_answer import make_chat_model
from ticket_store import TicketStore, local_actor
from trace_store import TraceStore, DEFAULT_TRACE_DB

BASE_DIR = Path(__file__).resolve().parent
STATIC = BASE_DIR / 'web'


# 每个Web会话启动一个stdio MCP子进程，并通过环境变量绑定知识库ID。
# 模型无法在search_docs参数中切换此ID；子进程的连接生命周期由Session.worker管理。
def default_client(knowledge_base='default'):
    return Client(StdioTransport(command=sys.executable,
        args=['-X', 'utf8', str(BASE_DIR / 'mcp_tools.py')], cwd=str(BASE_DIR),
        env={**os.environ, 'TICKET_KNOWLEDGE_BASE': knowledge_base}))


# 一个浏览器对话的服务端内存状态：独立历史、事件游标、消息队列和待确认草稿。
# 会话ID是随机令牌，用于定位本机会话；项目未实现真实账号登录或重启后的历史恢复。
class Session:
    # 为指定库创建独立消息历史并启动后台worker。库名称按展示数据放入提示，
    # 不作为权限来源；真正的检索隔离依赖MCP进程绑定与Chroma collection。
    def __init__(self, client_factory, model_factory, store_factory, actor_factory, knowledge_base=None, confirmation_timeout=300):
        self.id = secrets.token_urlsafe(24)
        self.events = []
        self.ready = False
        self.busy = False
        self.failed = False
        self.closing = False
        self.confirmation_timeout = confirmation_timeout
        # HTTP准入时清除，run_turn的finally置位；删除待审批会话时据此等待工具历史收尾。
        self.turn_finished = asyncio.Event()
        self.turn_finished.set()
        self.pending_draft = None
        self.decision = None
        self.queue = asyncio.Queue()
        self.factories = client_factory, model_factory, store_factory, actor_factory
        self.knowledge_base = dict(knowledge_base or DEFAULT_BASE)
        self.messages = initial_messages()
        if self.knowledge_base['id'] != 'default':
            role, policy = self.messages[0]
            self.messages[0] = (role, policy.replace('你是虚构产品知库助手 1.0 的技术支持助手。', '你是当前知识库的资料与技术支持助手。'))
        self.messages.append(('system', '当前知识库名称仅为展示数据：' + json.dumps(self.knowledge_base['name'], ensure_ascii=False)
                              + '。回答只使用当前search_docs返回的资料。禁止借用其他知识库资料补全；空库时说明没有资料。'))
        # MCP连接的进入、使用和退出都由同一后台任务负责。
        self.task = asyncio.create_task(self.worker())

    # 给本会话事件分配从1递增的ID并追加内存列表，前端可用after增量拉取。
    # 事件未持久化；页面刷新可重放，服务进程重启后不能恢复这份列表。
    async def emit(self, event):
        self.events.append({'id': len(self.events) + 1, **event})

    # 把草稿发送前端并await一个Future，暂停当前Agent轮次直到确认接口给出布尔结果。
    # finally清理等待状态；此函数只收集选择，真正的approve_draft在run_turn执行。
    async def confirm(self, draft):
        self.pending_draft = draft
        self.decision = asyncio.get_running_loop().create_future()
        await self.emit({'type': 'draft', 'draft': draft})
        try:
            try:
                return await asyncio.wait_for(self.decision, timeout=self.confirmation_timeout)
            except TimeoutError:
                # 超时按拒绝处理：不批准草稿、不调用创建工具；返回False让run_turn正常补齐ToolMessage。
                await self.emit({'type': 'decision', 'draft_id': draft['draft_id'], 'approved': False, 'reason': 'expired'})
                await self.emit({'type': 'status', 'content': '草稿确认已超时，本次创建已取消，草稿未获批准。'})
                return False
        finally:
            self.pending_draft = None
            self.decision = None

    # 同一个异步任务完成MCP进入、工具发现、队列消费和连接退出，避免跨任务关闭连接。
    # 绑定工具后依次处理问题；单次失败发error和done，初始化失败则标记会话failed。
    async def worker(self):
        client_factory, model_factory, store_factory, actor_factory = self.factories
        try:
            load_dotenv(BASE_DIR / '.env')
            store, actor = store_factory(), actor_factory()
            async with client_factory(self.knowledge_base['id']) as client:
                tools = await client.list_tools()
                definitions = [{'type': 'function', 'function': {
                    'name': t.name, 'description': t.description or '', 'parameters': t.input_schema,
                }} for t in tools if t.name in ALLOWED_TOOLS]
                model = model_factory(os.getenv('CHAT_BASE_URL', '').strip()).bind_tools(definitions)
                messages = self.messages
                self.ready = True
                await self.emit({'type': 'ready'})
                while True:
                    question = await self.queue.get()
                    try:
                        await run_turn(client, model, messages, question, confirm_draft=self.confirm,
                                       on_event=self.emit, store=store, actor=actor,
                                       session_id=hashlib.sha256(self.id.encode()).hexdigest(),
                                       knowledge_base_id=self.knowledge_base['id'])
                    except Exception as exc:
                        await self.emit({'type': 'error', 'content': f'本次处理未完成（{type(exc).__name__}），请核对结果后重试。'})
                    finally:
                        self.busy = False
                        await self.emit({'type': 'done'})
                        self.turn_finished.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failed, self.ready, self.busy = True, False, False
            self.turn_finished.set()
            await self.emit({'type': 'error', 'content': f'服务连接失败（{type(exc).__name__}），请检查本机配置后新建对话。'})

    # 取消后台任务并等待它退出，确保stdio子进程和MCP上下文有机会清理。
    async def close(self):
        self.closing, self.ready = True, False
        # 仅等待人工决定的轮次可以安全拒绝；让run_turn处理False并写完取消工具结果，
        # 再取消空闲worker，避免直接CancelledError留下缺失的ToolMessage。
        if self.decision is not None and not self.decision.done():
            self.decision.set_result(False)
            await self.emit({'type': 'decision', 'draft_id': self.pending_draft['draft_id'], 'approved': False, 'reason': 'closed'})
            await self.turn_finished.wait()
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)


# 构造本地Starlette应用；factory参数支持测试替身，默认使用真实模型、MCP和存储。
# sessions和knowledge缓存只属于这个应用实例；importing协调上传、配置和问答互斥。
def create_app(*, client_factory=None, model_factory=None, store_factory=TicketStore, actor_factory=local_actor, knowledge_factory=default_library, catalog=None, settings=None, confirmation_timeout=300):
    sessions = {}
    knowledge = {}
    importing = False
    shutting_down = False
    # 同一事件循环内的生命周期锁：准入检查、注册、关闭快照必须同锁完成。
    # body解析、MCP关闭和入库均在锁外await；期间由importing/closing拒绝新的准入。
    lifecycle_lock = asyncio.Lock()
    catalog = catalog if catalog is not None else KnowledgeBases()
    settings = settings if settings is not None else ServiceSettings()

    # 按kb_id延迟创建上传服务并缓存，避免每次文件列表请求重复打开Chroma。
    # 切换服务配置后会清除此缓存，下一次按新Embedding身份重新打开。
    def library(kb_id):
        if kb_id not in knowledge:
            knowledge[kb_id] = knowledge_factory(kb_id)
        return knowledge[kb_id]

    # 从本机目录清单解析已存在知识库；未知ID返回404，不回退到默认库。
    def base_for(kb_id):
        try:
            return catalog.get(kb_id)
        except KeyError:
            raise HTTPException(404, '知识库不存在，请重新选择。')

    # 服务关闭时逐个结束会话后台任务，释放MCP子进程资源。
    @asynccontextmanager
    async def lifespan(app):
        nonlocal shutting_down
        try:
            yield
        finally:
            async with lifecycle_lock:
                shutting_down = True
                snapshot = tuple(sessions.values())
                sessions.clear()
            await asyncio.gather(*(s.close() for s in snapshot))

    # 有Origin头时检查与当前服务同源，拒绝外站浏览器发起的写操作。
    # 没有Origin头的本机客户端仍可调用；它不是登录认证，也不是完整访问控制。
    def same_origin(request):
        origin = request.headers.get('origin')
        if origin and origin != f'{request.url.scheme}://{request.url.netloc}':
            raise HTTPException(403, '请从本地页面操作。')

    # 按路由中的sid查本应用内存会话；失效返回404提示新建对话。
    def session_for(request):
        session = sessions.get(request.path_params['sid'])
        if session is None:
            raise HTTPException(404, '会话已失效，请新建对话。')
        return session

    # 解析JSON且要求顶层对象，格式错误统一为400；具体字段由各接口继续检查。
    async def payload(request):
        try:
            data = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise HTTPException(400, '请求格式不正确。')
        if not isinstance(data, dict):
            raise HTTPException(400, '请求格式不正确。')
        return data

    # 返回静态页面骨架，交互逻辑由/static/app.js加载；这里不嵌入服务凭证。
    async def home(request):
        return FileResponse(STATIC / 'index.html')

    async def run_records(request):
        """只查当前本机Actor的记录；不启动会话、模型、Embedding或MCP。

        身份来自可信actor_factory，不采用URL中的user_id/workspace_id/role。
        缺失和无权访问统一404；数据库故障503且不回传异常正文/私人路径。
        """
        same_origin(request)
        run_id = request.path_params.get('run_id')
        if run_id is not None and len(run_id) > 200:
            raise HTTPException(400, '执行编号过长，请核对run_id。')
        try:
            limit = int(request.query_params.get('limit', '50'))
        except ValueError:
            raise HTTPException(400, 'limit必须是1到100的整数。')
        if not 1 <= limit <= 100:
            raise HTTPException(400, 'limit必须是1到100的整数。')

        def read():
            actor = actor_factory()
            # 与TurnTrace默认选择同一记录库，也适用于注入临时TicketStore的测试。
            business_path = getattr(store_factory(), 'path', None)
            path = Path(business_path).with_name('traces.sqlite3') if business_path else DEFAULT_TRACE_DB
            repository = TraceStore(path)
            if run_id is None:
                return {'runs': repository.list_runs(actor, limit=limit)}
            try:
                return repository.get_run_detail(run_id, actor)
            except PermissionError:
                raise HTTPException(404, '执行记录不存在或当前身份无权访问。') from None

        try:
            # SQLite连接/锁等待放到线程池，查询期间不阻塞审批及其他网页请求。
            data = await run_in_threadpool(read)
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(503, '无法读取执行记录，请检查本机记录库后重试。') from None
        return JSONResponse(data, headers={'Cache-Control': 'no-store'})

    # 先检查知识库存在，再检查真实模型配置就绪；未配置返回409，不启动MCP。
    # 注入测试模型时跳过生产配置门槛。返回201和sid后，前端继续轮询ready事件。
    async def new_session(request):
        same_origin(request)
        data = await payload(request) if await request.body() else {}
        async with lifecycle_lock:
            # await body后重新检查，不能沿用读取前的importing结论。
            if importing or shutting_down:
                raise HTTPException(409, '服务正在更新或关闭，请稍后新建对话。')
            base = base_for(data.get('kb_id', 'default'))
            if model_factory is None and not settings.ready():
                raise HTTPException(409,'请先打开服务配置，填写聊天与Embedding的地址、模型名称和Key。')
            session = Session(client_factory or default_client, model_factory or make_chat_model, store_factory, actor_factory, base, confirmation_timeout)
            sessions[session.id] = session
        return JSONResponse({'session_id': session.id, 'knowledge_base': base}, status_code=201)

    # 按after游标返回新增事件和当前ready/busy/failed/待确认草稿。
    # 这是短轮询JSON接口，不是SSE；同一事件由前端按ID避免重复渲染。
    async def events(request):
        session = session_for(request)
        try:
            after = max(0, int(request.query_params.get('after', '0')))
        except ValueError:
            raise HTTPException(400, '事件游标格式不正确。')
        return JSONResponse({'ready': session.ready, 'busy': session.busy, 'failed': session.failed,
                             'knowledge_base': session.knowledge_base,
                             'pending_draft': session.pending_draft,
                             'events': [e for e in session.events if e['id'] > after]})

    # 验证问题长度和会话空闲状态，先标busy、发用户消息事件，再入队返回202。
    # 202只表示问题已接受，答案和工具结果随后由events接口传回。
    async def send(request):
        same_origin(request)
        data = await payload(request)
        text = data.get('message')
        if not isinstance(text, str) or not text.strip() or len(text) > 20000:
            raise HTTPException(400, '请输入问题，长度不超过20000字。')
        async with lifecycle_lock:
            # payload可能等待很久：重新取会话，防止已删除/换配置的旧对象继续收消息。
            session = session_for(request)
            if importing or shutting_down or session.closing or not session.ready or session.busy:
                raise HTTPException(409, '请等待当前处理完成，或先确认/取消草稿。')
            session.busy = True
            session.turn_finished.clear()
            await session.emit({'type': 'message', 'role': 'user', 'content': text.strip()})
            session.queue.put_nowait(text.strip())
        return JSONResponse({'accepted': True}, status_code=202)

    # 只接收布尔approved，并核对当前等待草稿的ID及Future状态，拒绝过期或重复确认。
    # set_result唤醒Agent等待，不在HTTP接口中直接调用create_ticket。
    async def confirm(request):
        same_origin(request)
        data = await payload(request)
        if not isinstance(data.get('approved'), bool):
            raise HTTPException(400, '确认选择必须为布尔值。')
        async with lifecycle_lock:
            session = session_for(request)
            if (session.closing or session.pending_draft is None or session.decision is None or session.decision.done()
                    or data.get('draft_id') != session.pending_draft['draft_id']):
                raise HTTPException(409, '这份草稿当前不在等待确认，请刷新查看最新状态。')
            session.decision.set_result(data['approved'])
            await session.emit({'type': 'decision', 'draft_id': data['draft_id'], 'approved': data['approved']})
        return JSONResponse({'accepted': True})

    # 空闲或等待审批的会话可删除；已进入模型/工具处理的轮次拒绝直接中断，避免未知写入。
    async def delete(request):
        same_origin(request)
        async with lifecycle_lock:
            if importing or shutting_down:
                raise HTTPException(409, '服务正在更新或关闭，请稍后再试。')
            session = session_for(request)
            waiting = session.decision is not None and not session.decision.done()
            if session.busy and not waiting:
                raise HTTPException(409, '请先完成当前处理；等待确认的草稿可直接关闭取消。')
            # 先撤销准入并取出，后续await close期间send/confirm不能再取得旧对象。
            session.closing, session.ready = True, False
            sessions.pop(session.id)
        await session.close()
        return JSONResponse({'deleted': True})

    # 返回指定库网页上传清单；默认服务尚未配置时返回空列表及configuration_required。
    # 读取与向量库初始化放线程池中，避免同步工作阻塞异步请求循环。
    async def knowledge_files(request):
        if importing:
            raise HTTPException(409, '知识正在入库，请稍后查看。')
        base = base_for(request.query_params.get('kb_id', 'default'))
        if knowledge_factory is default_library and not settings.ready():
            return JSONResponse({'files':[],'knowledge_base':base,'configuration_required':True})
        try:
            files = await run_in_threadpool(lambda: library(base['id']).list_files())
            return JSONResponse({'files': files, 'knowledge_base': base})
        except Exception:
            raise HTTPException(503, '无法读取知识库，请检查本机配置。')

    # 验证目标库与配置，单文件读取到上限加1字节用于判断超限；上传期间暂停其他操作。
    # 关闭空闲MCP以刷新其Chroma缓存，线程池中入库，finally恢复原会话worker。
    # 只有目标库成功新增资料时追加重新检索提示，消息历史本身仍保留。
    async def upload(request):
        nonlocal importing
        same_origin(request)
        base = base_for(request.query_params.get('kb_id', 'default'))
        if knowledge_factory is default_library and not settings.ready():
            raise HTTPException(409,'请先完成聊天与Embedding服务配置，再上传知识资料。')
        # 完整读取上传body后再准入；否则慢请求会长期占用全局更新状态。
        async with request.form(max_files=1, max_fields=0) as form:
            file = form.get('file')
            if not isinstance(file, UploadFile):
                raise HTTPException(400, '请选择一个知识文件。')
            content = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(content) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, '文件不能超过 10MB。')
            filename = file.filename
        async with lifecycle_lock:
            if importing or shutting_down or any(s.busy for s in sessions.values()):
                raise HTTPException(409, '请先结束当前问答、确认或取消草稿，再上传资料。')
            if knowledge_factory is default_library and not settings.ready():
                raise HTTPException(409, '请先完成聊天与Embedding服务配置，再上传知识资料。')
            importing = True
            snapshot = tuple(sessions.values())
            for session in snapshot:
                session.ready = False
        paused = []
        try:
            # 关闭空闲MCP进程，避免其本地Chroma缓存遗漏刚上传的向量。
            # 会话消息仍保留，重新连接不会清空对话或工单。
            for session in snapshot:
                await session.close()
                paused.append(session)
            try:
                result = await run_in_threadpool(lambda: library(base['id']).upload(filename, content))
            except UploadError as exc:
                raise HTTPException(400, str(exc))
            except Exception:
                raise HTTPException(503, '本次入库未确认，请检查本机知识库配置后重试。')
            for session in paused:
                if session.knowledge_base['id'] == base['id'] and not result['duplicate']:
                    session.messages.append(('system', '知识库刚刚更新。后续产品知识问题请重新调用search_docs检索，不复用历史检索结果。'))
            return JSONResponse(result, status_code=200 if result['duplicate'] else 201)
        finally:
            async with lifecycle_lock:
                for session in paused:
                    if not shutting_down and sessions.get(session.id) is session:
                        session.failed, session.closing = False, False
                        session.task = asyncio.create_task(session.worker())
                importing = False

    # GET列知识库；POST验证名称后创建清单记录。此接口不自动上传资料或切换聊天库。
    async def bases(request):
        if request.method == 'GET':
            return JSONResponse({'bases': catalog.list()})
        same_origin(request)
        data = await payload(request)
        try:
            item = catalog.create(data.get('name'))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return JSONResponse({'knowledge_base': item}, status_code=201)

    # 把预期HTTP错误统一包装为error字段，供前端api函数展示简洁提示。
    async def error_response(request, exc):
        headers = dict(exc.headers or {})
        if request.url.path == '/api/runs' or request.url.path.startswith('/api/runs/'):
            headers['Cache-Control'] = 'no-store'
        return JSONResponse({'error': str(exc.detail)}, status_code=exc.status_code, headers=headers)

    # GET仅返回脱敏public；POST在无忙碌会话时保存本机配置。
    # 保存成功关闭旧会话、清空模型相关缓存，要求新建会话使用新参数；不删除原工单和索引。
    async def service_config(request):
        nonlocal importing
        if request.method == 'GET':
            try:
                return JSONResponse(settings.public())
            except Exception:
                raise HTTPException(503,'无法读取本机服务配置，请检查当前Windows账户和配置文件。')
        same_origin(request)
        data = await payload(request)
        async with lifecycle_lock:
            if importing or shutting_down or any(s.busy for s in sessions.values()):
                raise HTTPException(409,'请先完成当前问答、确认或取消草稿，再保存服务配置。')
            importing = True
            snapshot = tuple(sessions.values())
        try:
            try:
                public = settings.save(data)
            except ValueError as exc:
                raise HTTPException(400,str(exc))
            except Exception:
                raise HTTPException(503,'未能保存本机服务配置，请检查当前Windows账户和文件权限。')
            await asyncio.gather(*(s.close() for s in snapshot))
            async with lifecycle_lock:
                for session in snapshot:
                    sessions.pop(session.id, None)
                knowledge.clear()
            return JSONResponse(public)
        finally:
            async with lifecycle_lock:
                importing = False

    return Starlette(lifespan=lifespan, routes=[
        Route('/', home), Route('/api/sessions', new_session, methods=['POST']),
        Route('/api/settings', service_config, methods=['GET','POST'], max_body_size=65536),
        Route('/api/runs', run_records), Route('/api/runs/{run_id}', run_records),
        Route('/api/knowledge', knowledge_files),
        Route('/api/knowledge/bases', bases, methods=['GET', 'POST']),
        Route('/api/knowledge/upload', upload, methods=['POST'], max_body_size=10 * 1024 * 1024 + 65536),
        Route('/api/sessions/{sid}/events', events),
        Route('/api/sessions/{sid}/messages', send, methods=['POST']),
        Route('/api/sessions/{sid}/confirm', confirm, methods=['POST']),
        Route('/api/sessions/{sid}', delete, methods=['DELETE']),
        Mount('/static', StaticFiles(directory=STATIC, check_dir=False)),
    ], exception_handlers={HTTPException: error_response})


app = create_app()

if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv('TICKET_WEB_PORT', '8765'))
    print(f'工单助手网页：http://127.0.0.1:{port}')
    uvicorn.run(app, host='127.0.0.1', port=port)
