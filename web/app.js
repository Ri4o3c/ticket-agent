/* 开发说明：前端只调用同源Web接口，不直接连接聊天、Embedding或MinerU。
 * API Key仅在用户填写后提交本机后端，不写入sessionStorage；DOM使用文本节点展示返回内容。 */
const $ = (s) => document.querySelector(s);
const messages = $('#messages'), scroller = $('#scroll-area'), input = $('#question');
const toolLabels = {search_docs:'查阅产品资料',prepare_ticket:'准备工单草稿',create_ticket:'登记工单',get_ticket:'查询工单进度',find_ticket_by_request:'核查创建结果'};
/* 开发说明：sid标识当前服务端会话；after是已展示事件游标；epoch使旧会话的轮询结果失效。
 * sessionStorage只在当前浏览器标签会话内保存sid和库ID，重启后端后sid可能失效。 */
let sid = sessionStorage.getItem('ticket-agent-session'), after = 0, epoch = 0;
let ready = false, busy = false, pending = null, posting = false, trace = null;
let uploading = false;
let activeBase = sessionStorage.getItem('ticket-agent-kb') || 'default', bases = [], switching = false, creatingBase = false;
let savingSettings = false;
/* 开发说明：sources保存当前对话检索原文，drafts保存草稿卡片DOM；resetView在切库/新会话时清空。
 * 这是展示缓存，不能用来判断工单当前状态或绕过服务端人工审批。 */
const sources = new Map(), drafts = new Map();
/* 开发说明：通用DOM构造器，业务文字使用textContent；返回数据即使包含HTML也作为文本显示。 */
function node(tag, cls, text) { const n=document.createElement(tag); if(cls)n.className=cls; if(text!==undefined)n.textContent=text; return n; }
/* 开发说明：从页面内SVG symbol引用图标，name由代码中的固定图标名传入。 */
function icon(name) { const n=document.createElementNS('http://www.w3.org/2000/svg','svg');const u=document.createElementNS(n.namespaceURI,'use');u.setAttribute('href','#i-'+name);n.append(u);return n; }
function banner(text='') { $('#banner').textContent=text;$('#banner').hidden=!text; }
function bottom() { scroller.scrollTop=scroller.scrollHeight; }
/* 开发说明：按连接、忙碌、提交、上传、配置保存等状态统一更新按钮；避免重复点击。
 * 这些禁用仅改善交互，服务端仍需独立校验忙碌状态、草稿ID和权限。 */
function controls() {
  input.disabled=!ready||busy||posting||uploading||savingSettings;
  $('#send').disabled=!ready||busy||posting||uploading||savingSettings||!input.value.trim();
  document.querySelectorAll('[data-new]').forEach(b=>b.disabled=busy||posting||uploading||switching||savingSettings);
  $('#upload-start').disabled=uploading||busy||posting||creatingBase||!$('#knowledge-file').files.length;
  $('#chat-kb').disabled=!ready||busy||posting||uploading||switching||savingSettings;
  $('#upload-kb').disabled=uploading||creatingBase;
  $('#create-kb').disabled=uploading||creatingBase;
  $('#kb-name').disabled=uploading||creatingBase;
  $('#use-kb').disabled=!ready||busy||posting||uploading||switching;
  $('#save-settings').disabled=busy||posting||uploading||savingSettings||switching;
  input.placeholder=pending?'请先确认或取消上方草稿…':busy?'正在处理你的问题…':'描述你的问题，或输入工单编号…';
}
function activity(text) { $('#activity').hidden=!text;$('#activity span:last-child').textContent=text; }
/* 开发说明：同源JSON接口封装：成功返回解析后的数据，非2xx抛带HTTP状态的Error。
 * 上传FormData另用fetch，不能套用这里默认的application/json请求头。 */
async function api(path, options={}) {
  const response=await fetch(path,{headers:{'Content-Type':'application/json'},...options});
  const data=await response.json();
  if(!response.ok) {const e=new Error(data.error||'请求未完成');e.status=response.status;throw e;}
  return data;
}
/* 开发说明：只识别加粗和KB引用的轻量格式，不是完整Markdown渲染器。
 * 普通文本用createTextNode插入，引用按钮打开本会话已有原文，不执行模型生成的HTML。 */
function format(container,text) {
  for(const part of String(text).split(/(\*\*[^*]+\*\*|\[KB-\d{3,}\])/g)) {
    if(part.startsWith('**')&&part.endsWith('**')) container.append(node('strong','',part.slice(2,-2)));
    else if(/^\[KB-\d{3,}\]$/.test(part)) {const key=part.slice(1,-1);const b=node('button','cite',key);b.type='button';b.title='查看引用原文';b.onclick=()=>showReferences(key);container.append(b);}
    else container.append(document.createTextNode(part));
  }
}
/* 开发说明：将用户或助手消息追加到对话；真正的messages历史保存在服务端Session内。 */
function addMessage(role,text) {
  $('#welcome').hidden=true;
  const row=node('div','message '+role), avatar=node('div','avatar');avatar.append(icon('chat'));
  const body=node('div','message-body');body.append(node('div','message-label',role==='user'?'你':'知库助手'));
  const content=node('div','message-content');format(content,text);body.append(content);row.append(avatar,body);messages.append(row);
  if(role==='user') {trace=null;$('#session-title').textContent=text;}
}
/* 开发说明：把工具步骤追加到可折叠过程列表；展示“完成”只按该工具的ok字段判断。 */
function traceRow(text,cls='') {
  if(!trace) {const details=node('details','trace');details.append(node('summary','','查看处理过程'));trace=node('div','trace-list');details.append(trace);messages.append(details);}
  trace.append(node('div','trace-row '+cls,text));
}
/* 开发说明：草稿卡展示待核对标题/描述，确认与取消按钮只提交选择，不自行写工单。 */
function draftCard(draft) {
  $('#welcome').hidden=true;
  const card=node('section','draft-card');card.setAttribute('aria-label','待确认的工单草稿');
  const head=node('div','card-head'), label=node('span');label.append(icon('ticket'),document.createTextNode('工单草稿'));
  const badge=node('span','badge','待你确认');head.append(label,badge);
  const body=node('div','card-content');body.append(node('h3','',draft.title),node('p','',draft.description),node('div','record-id','草稿编号 · '+draft.draft_id));
  const actions=node('div','draft-actions'), yes=node('button','primary'), no=node('button','secondary','取消本次创建');
  yes.append(icon('check'),document.createTextNode('确认并创建'));yes.type=no.type='button';
  yes.onclick=()=>decide(draft.draft_id,true);no.onclick=()=>decide(draft.draft_id,false);actions.append(yes,no);
  card.append(head,body,actions);messages.append(card);drafts.set(draft.draft_id,{badge,actions,yes,no});
}
/* 开发说明：提交精确draft_id和布尔选择，防止重复点击；后端确认后仍需等create_ticket结果。 */
async function decide(id,approved) {
  const card=drafts.get(id);card.yes.disabled=card.no.disabled=true;
  try {await api(`/api/sessions/${sid}/confirm`,{method:'POST',body:JSON.stringify({draft_id:id,approved})});banner();}
  catch(e) {banner(e.message);card.yes.disabled=card.no.disabled=false;}
}
/* 开发说明：按工具返回的数据绘制正式工单快照；再次询问实时状态仍应重新调用get_ticket。 */
function ticketCard(ticket) {
  const card=node('section','ticket-card'),head=node('div','card-head');
  head.append(node('span','ticket-id',ticket.ticket_id),node('span','badge done',ticket.status));
  const body=node('div','card-content');body.append(node('h3','',ticket.title),node('p','',ticket.description));card.append(head,body);messages.append(card);
}
/* 开发说明：只展示本会话实际检索过的原文；编号未出现时明确提示，不能凭引用补造资料。 */
function showReferences(active) {
  const container=$('#references');container.replaceChildren();
  if(!sources.size)container.append(node('p','reference-item','当前还没有可展示的检索资料。'));
  if(active&&!sources.has(active))container.append(node('p','reference-item','该编号未出现在此会话已检索的资料中。'));
  for(const [id,doc] of sources) {
    const card=node('section','reference-item'+(id===active?' active':''));card.dataset.kb=id;
    card.append(node('span','',id),node('h3','',doc.title||'知识资料'),node('p','',String(doc.text||'').replace(/^##[^\n]*\n*/,'')));container.append(card);
  }
  $('#reference-panel').hidden=false;
  if(active)container.querySelector(`[data-kb="${active}"]`)?.scrollIntoView({block:'nearest'});
}
/* 开发说明：消费后端结构化事件：round/tool_call展示进度，draft等待确认，decision展示人的选择。
 * tool_result才更新检索来源或正式工单卡；done只表示本次处理结束，不代表业务一定成功。 */
function handle(event) {
  if(event.type==='message') {addMessage(event.role,event.content);if(event.role==='assistant')activity('');}
  if(event.type==='round')activity('正在理解问题…');
  if(event.type==='tool_call') {const label=toolLabels[event.name]||'处理请求';activity('正在'+label+'…');traceRow(label);}
  if(event.type==='status')activity(event.content);
  if(event.type==='draft') {activity('等待你核对草稿');draftCard(event.draft);}
  if(event.type==='decision') {
    const card=drafts.get(event.draft_id);
    if(card){card.yes.disabled=card.no.disabled=true;card.badge.textContent=event.approved?'已确认':'已取消';card.badge.className='badge '+(event.approved?'done':'cancelled');card.actions.replaceChildren(node('span','decision-note',event.approved?'已记录确认，正在等待正式创建结果。':'本次创建已取消，未登记正式工单。'));}
  }
  if(event.type==='tool_result') {
    const data=event.data;traceRow((toolLabels[event.name]||'操作')+(data.cancelled?' · 已取消':data.skipped?' · 未执行':data.ok?' · 完成':' · 未完成'),data.ok?'success':'failure');
    if(event.name==='search_docs'&&data.ok) {
      for(const doc of data.results||[])if(doc.knowledge_id)sources.set(doc.knowledge_id,doc);
      const b=node('button','reference-toggle');b.type='button';b.append(icon('book'),document.createTextNode(`查看参考资料 · ${(data.results||[]).length} 条`));b.onclick=()=>showReferences();messages.append(b);
      if(!$('#reference-panel').hidden)showReferences();
    }
    if(data.ok&&data.ticket)ticketCard(data.ticket);
  }
  if(event.type==='error') {
    $('#welcome').hidden=true;const card=node('div','error-card',event.content);
    if(event.request_id) {card.append(node('code','','原请求 · '+event.request_id));const b=node('button','secondary','核查此请求');b.onclick=()=>{input.value=`请核查创建请求 ${event.request_id} 是否已经生成工单，不要新建。`;input.dispatchEvent(new Event('input'));input.focus();};card.append(b);}
    messages.append(card);activity('');
  }
  if(event.type==='done')activity('');
}
/* 开发说明：清空旧会话展示、引用与草稿控件，并把事件游标归零以便重放新会话事件。 */
function resetView() {messages.replaceChildren();sources.clear();drafts.clear();trace=null;after=0;$('#welcome').hidden=false;$('#reference-panel').hidden=true;$('#session-title').textContent='从一个问题开始';activity('');}
/* 开发说明：创建或恢复会话并每650ms短轮询增量事件。fresh时删除旧空闲会话后建立新会话。
 * 每次连接递增epoch；切库后的旧响应即使晚到，也不能覆盖新会话状态。 */
async function connect(fresh=false,target=activeBase) {
  const current=++epoch;ready=false;switching=true;controls();banner();
  try {
    if(fresh&&sid){try{await api(`/api/sessions/${sid}`,{method:'DELETE'});}catch(e){if(e.status!==404)throw e;}}
    if(fresh){sid=null;sessionStorage.removeItem('ticket-agent-session');}
    resetView();
    if(sid) {try {const data=await api(`/api/sessions/${sid}/events`);activeBase=data.knowledge_base?.id||'default';}catch(e){if(e.status===404)sid=null;else throw e;}}
    if(!sid){const data=await api('/api/sessions',{method:'POST',body:JSON.stringify({kb_id:target})});sid=data.session_id;activeBase=data.knowledge_base.id;sessionStorage.setItem('ticket-agent-session',sid);}
    sessionStorage.setItem('ticket-agent-kb',activeBase);renderBases();
    input.value='';input.style.height='';
  } catch(e) {banner(e.message||'无法连接本地服务');$('#chat-kb').value=activeBase;$('#connection').className='connection failed';$('#connection span').textContent='连接失败';return;}
  finally{switching=false;controls();}
  while(current===epoch) {
    try {
      const data=await api(`/api/sessions/${sid}/events?after=${after}`);
      if(current!==epoch)return;
      for(const event of data.events){if(event.id<=after)continue;handle(event);after=event.id;}
      ready=data.ready;busy=data.busy;pending=data.pending_draft;
      $('#connection').className='connection'+(data.failed?' failed':ready?' online':'');
      $('#connection span').textContent=data.failed?'连接失败':ready?'服务已连接':'正在准备服务';
      if(data.events.length)bottom();controls();
    }catch(e){if(current!==epoch)return;ready=false;controls();banner(e.status===404?'会话已失效，请新建对话。':'连接暂时中断，正在尝试恢复…');}
    await new Promise(resolve=>setTimeout(resolve,650));
  }
}
/* 开发说明：发送前先占用posting标志，收到202后等待事件显示结果；请求失败保留问题供核查。 */
$('#composer').addEventListener('submit',async event=>{
  event.preventDefault();const text=input.value.trim();if(!text||!ready||busy||posting||uploading||savingSettings)return;
  posting=true;controls();
  try {await api(`/api/sessions/${sid}/messages`,{method:'POST',body:JSON.stringify({message:text})});input.value='';input.style.height='';busy=true;banner();}
  catch(e){banner(e.message||'发送未完成，请检查连接。');}
  finally{posting=false;controls();}
});
input.addEventListener('input',()=>{input.style.height='auto';input.style.height=Math.min(input.scrollHeight,170)+'px';controls();});
/* 开发说明：Enter发送，Shift+Enter换行；isComposing防止中文输入法选词时误发送。 */
input.addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();$('#composer').requestSubmit();}});
/* 开发说明：三条快捷提示只填入输入框并聚焦，用户仍需发送；不会自动调用工具。 */
document.querySelectorAll('[data-prompt]').forEach(b=>b.onclick=()=>{if(busy||!ready||uploading||switching)return;input.value=b.dataset.prompt;input.dispatchEvent(new Event('input'));input.focus();});
document.querySelectorAll('[data-new]').forEach(b=>b.onclick=()=>connect(true));
$('#close-references').onclick=()=>$('#reference-panel').hidden=true;
function uploadFeedback(text,kind='') {
  const box=$('#upload-feedback');box.hidden=!text;box.className='upload-feedback '+kind;box.textContent=text;
}
/* 开发说明：列表请求绑定当次选择的目标库；响应返回后再次检查选择，忽略已切库的旧结果。 */
async function loadKnowledge() {
  const target=$('#upload-kb').value;
  const list=$('#knowledge-files');list.replaceChildren(node('p','knowledge-empty','正在读取资料…'));
  try {
    const data=await api('/api/knowledge?kb_id='+encodeURIComponent(target));if($('#upload-kb').value!==target)return;list.replaceChildren();
    if(!data.files.length)list.append(node('p','knowledge-empty','还没有从网页上传资料，先选择一份文件开始。'));
    for(const file of data.files) {
      const row=node('div','knowledge-file-row'),mark=node('div','file-mark');mark.append(icon('book'));
      const body=node('div','file-info');body.append(node('strong','',file.filename),node('span','',`${file.chunks} 个分块 · ${Math.max(1,Math.ceil(file.size/1024))} KB · ${new Date(file.created_at).toLocaleString('zh-CN')}`));
      row.append(mark,body,node('span','badge done','可检索'));list.append(row);
    }
  }catch(e){if($('#upload-kb').value===target)list.replaceChildren(node('p','knowledge-empty',e.message||'暂时无法读取知识库，请稍后刷新。'));}
}
/* 开发说明：同一清单渲染聊天库与上传库；两者可以暂时不同，上传目标不会自动改变当前会话。 */
function renderBases() {
  const uploadTarget=$('#upload-kb').value;
  for(const selector of ['#chat-kb','#upload-kb']) {
    const select=$(selector);select.replaceChildren();
    for(const base of bases){const option=node('option','',base.name);option.value=base.id;select.append(option);}
    select.value=selector==='#chat-kb'?activeBase:(bases.some(b=>b.id===uploadTarget)?uploadTarget:activeBase);
  }
}
async function loadBases() {const data=await api('/api/knowledge/bases');bases=data.bases;if(!bases.some(b=>b.id===activeBase))activeBase='default';renderBases();}
$('#open-knowledge').onclick=async()=>{$('#knowledge-dialog').showModal();try{await loadBases();$('#upload-kb').value=activeBase;uploadFeedback('');loadKnowledge();}catch(e){uploadFeedback(e.message,'error');}};
$('#close-knowledge').onclick=()=>$('#knowledge-dialog').close();
$('#refresh-knowledge').onclick=()=>{if(!uploading)loadKnowledge();};
$('#upload-kb').onchange=()=>{uploadFeedback('');loadKnowledge();};
/* 开发说明：真正切换聊天库时建立新会话，配合后端MCP绑定实现检索隔离。 */
$('#chat-kb').onchange=()=>{const target=$('#chat-kb').value;if(target!==activeBase)connect(true,target);};
$('#use-kb').onclick=()=>{const target=$('#upload-kb').value;if(target!==activeBase)connect(true,target);$('#knowledge-dialog').close();};
/* 开发说明：创建成功后只把上传目标指向新库；“使用此库聊天”才切换对话上下文。 */
$('#create-kb-form').onsubmit=async event=>{
  event.preventDefault();const name=$('#kb-name').value.trim();if(!name||creatingBase||uploading)return;
  creatingBase=true;controls();
  try{const data=await api('/api/knowledge/bases',{method:'POST',body:JSON.stringify({name})});await loadBases();$('#upload-kb').value=data.knowledge_base.id;$('#kb-name').value='';uploadFeedback(`已创建“${data.knowledge_base.name}”，现在可以向此库上传资料。`,'success');await loadKnowledge();}
  catch(e){uploadFeedback(e.message||'创建知识库未完成，请刷新后核查。','error');}
  finally{creatingBase=false;controls();}
};
$('#knowledge-file').onchange=()=>{const file=$('#knowledge-file').files[0];$('#selected-file').textContent=file?file.name:'选择一份知识资料';uploadFeedback('');controls();};
/* 开发说明：以FormData上传单个文件，由后端解析、分块、向量化并核对完成。
 * 浏览器限制只是提前提示；后端还有独立大小/类型校验，入库成功由返回结果确认。 */
$('#upload-form').onsubmit=async event=>{
  event.preventDefault();const file=$('#knowledge-file').files[0];if(!file||busy||posting||uploading||creatingBase)return;
  if(file.size>10*1024*1024){uploadFeedback('文件不能超过 10MB，请拆分后上传。','error');return;}
  uploading=true;controls();$('#knowledge-file').disabled=true;$('#refresh-knowledge').disabled=true;
  $('#upload-start').textContent='正在入库…';uploadFeedback('正在解析、分块并生成向量，请等待入库结果…','working');
  try {
    const form=new FormData();form.append('file',file);
    const response=await fetch('/api/knowledge/upload?kb_id='+encodeURIComponent($('#upload-kb').value),{method:'POST',body:form});const data=await response.json();
    if(!response.ok)throw new Error(data.error||'上传未完成');
    uploadFeedback(data.duplicate?`已跳过：这份内容已入库（${data.filename}）。`:`${data.filename} 已入库，共 ${data.chunks} 个分块，可以开始提问。${data.warnings?.length?' 提示：'+data.warnings.join('；'):''}`,'success');
    $('#knowledge-file').value='';$('#selected-file').textContent='选择一份知识资料';await loadKnowledge();
  }catch(e){uploadFeedback(e.message||'未收到入库结果，请恢复连接后刷新列表核查；重复上传会自动去重。','error');}
  finally{uploading=false;$('#knowledge-file').disabled=false;$('#refresh-knowledge').disabled=false;$('#upload-start').textContent='上传并入库';controls();}
};
/* 开发说明：启动先加载库清单和公开配置；缺配置时打开空白表单，不创建模型会话。
 * configured只代表参数完整，真实API是否可用要在后续实际调用中确认。 */
async function startApplication() {
  try {
    await loadBases();
    const config=await api('/api/settings');
    if(!config.chat.configured||!config.embedding.configured) {
      ready=false;controls();$('#connection span').textContent='待配置';
      banner('首次使用：请先填写自己的聊天和Embedding接口，保存后即可开始。');
      $('#settings-dialog').showModal();showSettings(config);
      settingsFeedback('此版本不提供默认接口、模型或Key，请填写后保存。');
      return;
    }
    connect();
  }catch(e){banner(e.message||'无法读取初始配置');$('#connection span').textContent='连接失败';}
}
startApplication();

function settingsFeedback(text,kind='') {const box=$('#settings-feedback');box.textContent=text;box.hidden=!text;box.className='upload-feedback '+kind;}
/* 开发说明：仅回填地址、模型、维度和Key状态；Key输入框始终清空，后端不返回明文。
 * 已有Key时留空保留，首次必填；MinerU保持可选且仅配置，不触发解析。 */
function showSettings(data) {
  for(const section of ['chat','embedding','mineru']) {
    $('#'+section+'-base-url').value=data[section].base_url||'';
    if(section!=='mineru')$('#'+section+'-model').value=data[section].model;
    $('#'+section+'-key').value='';
    $('#'+section+'-key').required=section!=='mineru'&&!data[section].has_key;
    $('#'+section+'-key').placeholder=data[section].has_key?'留空保留当前Key':section==='mineru'?'填写自己的Token（可选）':'填写自己的API Key';
    $('#'+section+'-key-state').textContent=data[section].has_key?'Key 已配置':'Key 未配置';
  }
  $('#embedding-dimensions').value=data.embedding.dimensions??'';
}
$('#open-settings').onclick=async()=>{
  $('#settings-dialog').showModal();settingsFeedback('正在读取本机配置…','working');
  try{showSettings(await api('/api/settings'));settingsFeedback('');}
  catch(e){settingsFeedback(e.message||'无法读取配置，请稍后重试。','error');}
};
/* 开发说明：关闭或取消对话框时清掉页面密码输入；不会删除后端已经加密保存的Key。 */
$('#close-settings').onclick=()=>{if(!savingSettings){$('#settings-dialog').close();document.querySelectorAll('#settings-form input[type="password"]').forEach(i=>i.value='');}};
$('#settings-dialog').addEventListener('cancel',event=>{if(savingSettings)event.preventDefault();else document.querySelectorAll('#settings-form input[type="password"]').forEach(i=>i.value='');});
/* 开发说明：提交用户显式填写的独立服务参数，空维度转null。保存成功后清掉旧sid再建会话，
 * 使新模型和索引配置生效；保存请求失败时提示核查，不假定原配置已被替换。 */
$('#settings-form').onsubmit=async event=>{
  event.preventDefault();if(busy||posting||uploading||savingSettings||switching)return;
  const payload={};for(const section of ['chat','embedding','mineru']) {
    payload[section]={base_url:$('#'+section+'-base-url').value.trim(),api_key:$('#'+section+'-key').value};
    if(section!=='mineru')payload[section].model=$('#'+section+'-model').value.trim();
  }
  const dimensions=$('#embedding-dimensions').value;payload.embedding.dimensions=dimensions?Number(dimensions):null;
  savingSettings=true;controls();$('#close-settings').disabled=true;settingsFeedback('正在保存本机配置…','working');
  try {
    const data=await api('/api/settings',{method:'POST',body:JSON.stringify(payload)});showSettings(data);
    settingsFeedback('配置已保存，正在开启新会话。MinerU 参数已留存，解析流程尚未启用。','success');
    sid=null;sessionStorage.removeItem('ticket-agent-session');await loadBases();
    // connect持续轮询，因此不等待其结束。
    connect(true);
  }catch(e){settingsFeedback(e.message||'未收到保存结果，请重新打开配置核查。','error');}
  finally{savingSettings=false;$('#close-settings').disabled=false;controls();}
};
