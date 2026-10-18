"""本地网页入口：python web_app.py，浏览器访问 http://127.0.0.1:8765。"""
import asyncio
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
    def __init__(self, client_factory, model_factory, store_factory, actor_factory, knowledge_base=None):
        self.id = secrets.token_urlsafe(24)
        self.events = []
        self.ready = False
        self.busy = False
        self.failed = False
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
            return await self.decision
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
                                       on_event=self.emit, store=store, actor=actor)
                    except Exception as exc:
                        await self.emit({'type': 'error', 'content': f'本次处理未完成（{type(exc).__name__}），请核对结果后重试。'})
                    finally:
                        self.busy = False
                        await self.emit({'type': 'done'})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failed, self.ready, self.busy = True, False, False
            await self.emit({'type': 'error', 'content': f'服务连接失败（{type(exc).__name__}），请检查本机配置后新建对话。'})

    # 取消后台任务并等待它退出，确保stdio子进程和MCP上下文有机会清理。
    async def close(self):
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)


# 构造本地Starlette应用；factory参数支持测试替身，默认使用真实模型、MCP和存储。
# sessions和knowledge缓存只属于这个应用实例；importing协调上传、配置和问答互斥。
def create_app(*, client_factory=None, model_factory=None, store_factory=TicketStore, actor_factory=local_actor, knowledge_factory=default_library, catalog=None, settings=None):
    sessions = {}
    knowledge = {}
    importing = False
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
        yield
        await asyncio.gather(*(s.close() for s in sessions.values()))

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

    # 先检查知识库存在，再检查真实模型配置就绪；未配置返回409，不启动MCP。
    # 注入测试模型时跳过生产配置门槛。返回201和sid后，前端继续轮询ready事件。
    async def new_session(request):
        same_origin(request)
        if importing:
            raise HTTPException(409, '知识正在入库，请稍后新建对话。')
        data = await payload(request) if await request.body() else {}
        base = base_for(data.get('kb_id', 'default'))
        if model_factory is None and not settings.ready():
            raise HTTPException(409,'请先打开服务配置，填写聊天与Embedding的地址、模型名称和Key。')
        session = Session(client_factory or default_client, model_factory or make_chat_model, store_factory, actor_factory, base)
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
        session = session_for(request)
        data = await payload(request)
        text = data.get('message')
        if not isinstance(text, str) or not text.strip() or len(text) > 20000:
            raise HTTPException(400, '请输入问题，长度不超过20000字。')
        if importing or not session.ready or session.busy:
            raise HTTPException(409, '请等待当前处理完成，或先确认/取消草稿。')
        session.busy = True
        await session.emit({'type': 'message', 'role': 'user', 'content': text.strip()})
        session.queue.put_nowait(text.strip())
        return JSONResponse({'accepted': True}, status_code=202)

    # 只接收布尔approved，并核对当前等待草稿的ID及Future状态，拒绝过期或重复确认。
    # set_result唤醒Agent等待，不在HTTP接口中直接调用create_ticket。
    async def confirm(request):
        same_origin(request)
        session = session_for(request)
        data = await payload(request)
        if not isinstance(data.get('approved'), bool):
            raise HTTPException(400, '确认选择必须为布尔值。')
        if (session.pending_draft is None or session.decision is None or session.decision.done()
                or data.get('draft_id') != session.pending_draft['draft_id']):
            raise HTTPException(409, '这份草稿当前不在等待确认，请刷新查看最新状态。')
        session.decision.set_result(data['approved'])
        await session.emit({'type': 'decision', 'draft_id': data['draft_id'], 'approved': data['approved']})
        return JSONResponse({'accepted': True})

    # 只允许删除空闲会话；先关闭后台MCP再移除内存项，不删除SQLite工单或知识资料。
    async def delete(request):
        same_origin(request)
        if importing:
            raise HTTPException(409, '知识正在入库，请稍后再试。')
        session = session_for(request)
        if session.busy:
            raise HTTPException(409, '请先完成或取消当前操作。')
        await session.close()
        sessions.pop(session.id, None)
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
        if importing or any(s.busy for s in sessions.values()):
            raise HTTPException(409, '请先结束当前问答、确认或取消草稿，再上传资料。')
        importing = True
        paused = []
        try:
            async with request.form(max_files=1, max_fields=0) as form:
                file = form.get('file')
                if not isinstance(file, UploadFile):
                    raise HTTPException(400, '请选择一个知识文件。')
                content = await file.read(MAX_UPLOAD_BYTES + 1)
                if len(content) > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, '文件不能超过 10MB。')
                filename = file.filename
            # 关闭空闲MCP进程，避免其本地Chroma缓存遗漏刚上传的向量。
            # 会话消息仍保留，重新连接不会清空对话或工单。
            for session in sessions.values():
                session.ready = False
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
            for session in paused:
                session.failed = False
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
        return JSONResponse({'error': str(exc.detail)}, status_code=exc.status_code)

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
        if importing or any(s.busy for s in sessions.values()):
            raise HTTPException(409,'请先完成当前问答、确认或取消草稿，再保存服务配置。')
        importing = True
        try:
            try:
                public = settings.save(data)
            except ValueError as exc:
                raise HTTPException(400,str(exc))
            except Exception:
                raise HTTPException(503,'未能保存本机服务配置，请检查当前Windows账户和文件权限。')
            await asyncio.gather(*(s.close() for s in sessions.values()))
            sessions.clear()
            knowledge.clear()
            return JSONResponse(public)
        finally:
            importing = False

    return Starlette(lifespan=lifespan, routes=[
        Route('/', home), Route('/api/sessions', new_session, methods=['POST']),
        Route('/api/settings', service_config, methods=['GET','POST'], max_body_size=65536),
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
