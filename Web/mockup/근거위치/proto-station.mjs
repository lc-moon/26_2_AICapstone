// 시안 3: 선택한 곳의 근거(가까운 측정소·CCTV) 위치. ?proto=station 일 때만.
// - 지도 시점은 건드리지 않는다(초점은 선택한 곳)
// - 지도 위에 "서 있는 핀"(막대 + 땅 위 퍼지는 점)으로, 지도 아이콘과 구분되게. 선택 중에는 지도 장소 아이콘을 흐리게
// - 실제로 가리는 것(패널·카드·버튼)에 가려지거나 화면 밖이면 가장자리 방향 핀
// - 핀을 누르면 선택한 곳은 화면 같은 자리에 둔 채, 목표가 보일 만큼만 물러난다. 다시 누르거나 ×면 원래 시점으로
// - &line=1 이면 선택한 곳과 잇는 옅은 선(비교용)
import {gradeOf,validValue,rgb,formatTime} from './core.mjs';
import {photoURL} from './evidence.mjs';
let stations=[],cameras=[],targets=[],overlay=null,raf=0,mapRef=null,saved=null,active=null,ctx={},dimmed=false,measuring=false;
const empty={type:'FeatureCollection',features:[]};
const ICON={
 station:'<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M4 17c2-3 4-3 6 0s4 3 6 0 3-2 4-1"/><path d="M4 10c2-3 4-3 6 0s4 3 6 0 3-2 4-1"/></svg>',
 cctv:'<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7l13 4-2 5L2 12z"/><path d="M16 11l4-1v5l-5-1"/><path d="M6 13v5H3"/></svg>'};
const COLOR={station:'#1d5c8f',cctv:'#c8323b'},NAME={station:'측정소',cctv:'CCTV'};
const css=`
.ev-pin{position:relative;width:0;height:0}
.ev-head{position:absolute;left:0;margin-left:var(--dx,0px);bottom:30px;transform:translateX(-50%);display:flex;align-items:center;gap:6px;border:0;padding:4px 12px 4px 4px;border-radius:22px;background:var(--c);color:#fff;font:700 13px/1 inherit;white-space:nowrap;cursor:pointer;box-shadow:0 0 0 2.5px #fff,0 6px 16px #0c2a2e66;animation:ev-rise .45s cubic-bezier(.2,.9,.3,1.3) both}
.ev-head i{display:grid;place-items:center;width:28px;height:28px;border-radius:50%;background:#fff;color:var(--c)}
.ev-head:hover{filter:brightness(1.08)}.ev-pin.on .ev-head{box-shadow:0 0 0 3px #fff,0 0 0 6px var(--c),0 6px 16px #0c2a2e66}
.ev-stem{position:absolute;left:-1.5px;bottom:2px;width:3px;height:30px;background:linear-gradient(var(--c),#fff0);border-radius:2px}
.ev-ground{position:absolute;left:-6px;top:-6px;width:12px;height:12px;border-radius:50%;background:var(--c);border:2px solid #fff;box-sizing:border-box}
.ev-ground:before{content:"";position:absolute;inset:-2px;border-radius:50%;border:2px solid var(--c);animation:ev-pulse 2s ease-out infinite}
.ev-detail{position:absolute;left:0;bottom:78px;transform:translateX(-50%);background:#fff;border-radius:12px;padding:12px 14px;width:228px;box-shadow:0 10px 30px #0c2a2e40;text-align:left;cursor:default;animation:ev-fade .3s ease both;z-index:3}
.ev-detail .k{font-size:10px;letter-spacing:1px;font-weight:700}.ev-detail .n{font-size:14px;font-weight:700;margin:2px 0 6px;color:#1f3437}
.ev-detail .v{font-size:22px;font-weight:700;color:#1f3437;letter-spacing:-.5px}.ev-detail .v small{font-size:11px;font-weight:400;margin-left:3px}
.ev-detail .g{display:inline-block;font-size:11px;border-radius:12px;padding:2px 8px;margin-left:6px;vertical-align:4px;color:#1f3437;letter-spacing:0}
.ev-detail .t{font-size:11px;color:#687b7e;margin-top:4px}
.ev-detail img{display:block;width:200px;height:auto;border-radius:6px;margin:6px 0 2px}
.ev-detail .x{position:absolute;top:6px;right:8px;border:0;background:none;font-size:16px;color:#687b7e;cursor:pointer}
.ev-edges{position:absolute;inset:0;pointer-events:none;z-index:2}
.ev-edge{position:absolute;pointer-events:auto;cursor:pointer;border:0;background:none;padding:0;transform:translate(-50%,-50%);animation:ev-fade .3s ease both}
.ev-edge svg{display:block;filter:drop-shadow(0 3px 6px #0c2a2e66)}
.ev-edge span{position:absolute;top:50%;transform:translateY(-50%);white-space:nowrap;background:var(--c);color:#fff;border-radius:12px;padding:4px 9px;font:700 12px/1.2 inherit;box-shadow:0 0 0 2px #fff,0 3px 10px #0c2a2e44}
@keyframes ev-pulse{from{transform:scale(1);opacity:.9}to{transform:scale(3.4);opacity:0}}
@keyframes ev-rise{from{opacity:0;transform:translate(-50%,10px)}to{opacity:1;transform:translate(-50%,0)}}
@keyframes ev-fade{from{opacity:0}to{opacity:1}}`;

export function initStationProto(map,options){
  mapRef=map;ctx=options;
  document.head.append(Object.assign(document.createElement('style'),{textContent:css}));
  overlay=Object.assign(document.createElement('div'),{className:'ev-edges'});map.getContainer().append(overlay);
  map.addSource('ev-link',{type:'geojson',data:empty});
  map.addLayer({id:'ev-link',type:'line',source:'ev-link',layout:{'line-cap':'round'},paint:{'line-color':['get','color'],'line-width':2.5,'line-opacity':.45}});
  void Promise.all([fetch('../shared/stations.json').then(r=>r.json()),fetch('../shared/cameras.json').then(r=>r.json())]).then(([s,c])=>{stations=s.stations;cameras=c.cameras;});
  map.on('move',()=>{if(measuring)return;cancelAnimationFrame(raf);raf=requestAnimationFrame(updateEdges);});
  addEventListener('resize',()=>updateEdges());
}

function km(a,b){const R=6371,r=Math.PI/180,dl=(b[1]-a[1])*r,dn=(b[0]-a[0])*r,x=Math.sin(dl/2)**2+Math.cos(a[1]*r)*Math.cos(b[1]*r)*Math.sin(dn/2)**2;return 2*R*Math.asin(Math.sqrt(x));}

// 선택 중에는 지도의 장소 아이콘·글자를 흐리게(근거 핀이 묻히지 않게)
function dim(on){
  if(!mapRef||dimmed===on)return;dimmed=on;
  for(const l of mapRef.getStyle().layers)if(l.type==='symbol'&&/^poi/.test(l.id)){mapRef.setPaintProperty(l.id,'icon-opacity',on?.25:1);mapRef.setPaintProperty(l.id,'text-opacity',on?.3:1);}
}

export function clearStation(){
  targets.forEach(t=>{t.marker.remove();t.edge?.remove();});targets=[];active=null;saved=null;dim(false);
  overlay?.replaceChildren();mapRef?.getSource('ev-link')?.setData(empty);
}

function links(){
  const show=new URLSearchParams(location.search).get('line')==='1';
  const list=show?targets:active?[active]:[];
  mapRef.getSource('ev-link').setData({type:'FeatureCollection',features:list.map(t=>({type:'Feature',properties:{color:COLOR[t.type]},geometry:{type:'LineString',coordinates:[t.point,t.at]}}))});
}

export function showStation(map,point,data){
  clearStation();
  const s=data.station&&stations.find(x=>x.station_name===data.station.name);
  if(s)targets.push({type:'station',at:[s.lon,s.lat],km:data.station.distance_km,info:data.station});
  // 실서버는 아직 camera(S3)가 없어 cameras.json에서 가장 가까운 카메라로 대신한다
  let cam=data.camera&&cameras.find(c=>c.camera_id===String(data.camera.camera_id));
  if(!cam&&cameras.length)cam=cameras.reduce((a,c)=>km(point,[c.lon,c.lat])<km(point,[a.lon,a.lat])?c:a);
  if(cam)targets.push({type:'cctv',at:[cam.lon,cam.lat],km:data.camera?.distance_km??data.estimate?.nearest_cctv_km??km(point,[cam.lon,cam.lat]),info:{...cam,...(data.camera||{})}});
  for(const t of targets){
    const el=document.createElement('div');el.className='ev-pin';el.style.setProperty('--c',COLOR[t.type]);
    el.innerHTML=`<i class="ev-stem"></i><i class="ev-ground"></i><button type="button" class="ev-head"><i>${ICON[t.type]}</i>${NAME[t.type]}</button>`;
    el.querySelector('.ev-head').onclick=e=>{e.stopPropagation();toggle(t);};
    t.el=el;t.point=point;
    t.marker=new maplibregl.Marker({element:el,anchor:'center'}).setLngLat(t.at).addTo(map);
  }
  if(targets.length)dim(true);
  links();updateEdges();
}

// 실제로 화면을 가리는 것들(지도 기준 좌표)
function blockers(){
  const c=mapRef.getContainer().getBoundingClientRect();
  return ['#panel','#title','#place-search','#legend','#status','.tools','.maplibregl-ctrl-top-right','#toast']
    .map(s=>document.querySelector(s)).filter(e=>e&&!e.hidden&&e.getClientRects().length>0)
    .map(e=>{const r=e.getBoundingClientRect();return {l:r.left-c.left-8,t:r.top-c.top-8,r:r.right-c.left+8,b:r.bottom-c.top+8};});
}
function free(x,y,B,m=24,pad=0){
  const c=mapRef.getContainer();
  return x>=m&&y>=m&&x<=c.clientWidth-m&&y<=c.clientHeight-m&&!B.some(b=>x>=b.l-pad&&x<=b.r+pad&&y>=b.t-pad&&y<=b.b+pad);
}
// 땅 위 점과 핀 머리(땅 위 30~66px, 폭 약 100px)가 다 가려지지 않아야 보인다고 친다
const seen=(q,B,m,pad=0)=>[[0,0],[0,-68],[-52,-48],[52,-48],[-52,-32],[52,-32]].every(([dx,dy])=>free(q.x+dx,q.y+dy,B,m,pad));

function updateEdges(){
  if(!mapRef||!overlay||measuring)return;
  const B=blockers(),edges=[],c=mapRef.getContainer();
  for(const t of targets){
    const q=mapRef.project(t.at),ok=seen(q,B,16);
    t.el.style.visibility=ok?'visible':'hidden';
    if(ok)continue;
    // 선택한 곳에서 목표 쪽으로 쏜 선 위에서, 목표에 가장 가까우면서 가려지지 않은 자리
    let o=mapRef.project(t.point);if(!free(o.x,o.y,B))o={x:c.clientWidth/2,y:c.clientHeight/3};
    const dx=q.x-o.x,dy=q.y-o.y,m=30;
    const sx=dx>0?(c.clientWidth-m-o.x)/dx:dx<0?(m-o.x)/dx:Infinity,sy=dy>0?(c.clientHeight-m-o.y)/dy:dy<0?(m-o.y)/dy:Infinity;
    let s=Math.min(1,sx,sy);
    // 핀 크기(약 20px 반경)만큼 가리는 것에서 떨어진 자리
    const clear=(x,y)=>free(x,y,B,m)&&free(x-20,y,B,m)&&free(x+20,y,B,m)&&free(x,y-22,B,m)&&free(x,y+22,B,m);
    while(s>0&&!clear(o.x+dx*s,o.y+dy*s))s-=.01;
    edges.push({t,x:o.x+dx*Math.max(s,0),y:o.y+dy*Math.max(s,0),angle:Math.atan2(dy,dx)*180/Math.PI});
  }
  // 두 핀 머리가 겹치면 머리만 좌우로 벌린다(땅 위 점은 제자리)
  const shown=targets.filter(t=>t.el.style.visibility!=='hidden');
  targets.forEach(t=>t.el.style.setProperty('--dx','0px'));
  if(shown.length===2){const [p,q]=shown.map(t=>mapRef.project(t.at));
    if(Math.abs(p.x-q.x)<110&&Math.abs(p.y-q.y)<44){const d=(110-Math.abs(p.x-q.x))/2,left=p.x<=q.x?0:1;shown.forEach((t,i)=>t.el.style.setProperty('--dx',(i===left?-d:d)+'px'));}}
  // 가장자리 핀은 한 번만 만들고 위치만 옮긴다(매 프레임 새로 만들면 페이드가 계속 다시 시작됨)
  for(const t of targets){
    const e=edges.find(x=>x.t===t);
    if(!e){if(t.edge)t.edge.hidden=true;continue;}
    if(!t.edge){
      const b=document.createElement('button');b.type='button';b.className='ev-edge';b.style.setProperty('--c',COLOR[t.type]);
      b.title=`${NAME[t.type]} 위치 보기`;b.setAttribute('aria-label',`${NAME[t.type]} 위치 보기`);
      b.innerHTML=`<svg width="34" height="43" viewBox="0 0 30 38"><path d="M15 37C9 27 1 21 1 14a14 14 0 0 1 28 0c0 7-8 13-14 23z" fill="${COLOR[t.type]}" stroke="#fff" stroke-width="1.8"/><circle cx="15" cy="14" r="6" fill="#fff"/></svg><span></span>`;
      b.querySelector('span').textContent=`${NAME[t.type]} ${Number.isFinite(t.km)?t.km.toFixed(1)+'km':''}`;
      b.onclick=ev=>{ev.stopPropagation();toggle(t);};
      overlay.append(b);t.edge=b;
    }
    const b=t.edge;b.hidden=false;b.style.left=e.x+'px';b.style.top=e.y+'px';
    b.querySelector('svg').style.transform=`rotate(${e.angle-90}deg)`;
    const span=b.querySelector('span'),right=e.x>c.clientWidth/2;span.style.right=right?'38px':'';span.style.left=right?'':'38px';
  }
}

function detail(t){
  const box=document.createElement('div');box.className='ev-detail';box.onclick=e=>e.stopPropagation();
  const x=Object.assign(document.createElement('button'),{className:'x',textContent:'×',type:'button',title:'선택한 곳으로'});x.onclick=e=>{e.stopPropagation();toggle(t);};
  const k=Object.assign(document.createElement('div'),{className:'k',textContent:t.type==='station'?'가까운 측정소 · 실측':'가까운 CCTV · 추정에 쓴 사진'});k.style.color=COLOR[t.type];
  box.append(x,k,Object.assign(document.createElement('div'),{className:'n',textContent:t.info.name}));
  if(t.type==='cctv'){const src=photoURL(t.info.photo_url||`/cam/${t.info.camera_id}.jpg`,ctx);if(src){const img=document.createElement('img');img.src=src;img.alt=`${t.info.name} 사진`;img.onerror=()=>img.remove();box.append(img);}}
  if(validValue(t.info.pm25)){const g=gradeOf(t.info.pm25),v=document.createElement('div');v.className='v';v.innerHTML=`${Math.round(t.info.pm25)}<small>㎍/㎥</small><span class="g"></span>`;v.querySelector('.g').textContent=g.name;v.querySelector('.g').style.background=rgb(g.light);box.append(v);}
  const when=t.type==='station'?`${formatTime(t.info.data_time)} 기준 · ${t.info.period||''}`:t.info.captured_at?`${formatTime(t.info.captured_at)} 촬영`:'최근 낮 회차 사진';
  box.append(Object.assign(document.createElement('div'),{className:'t',textContent:`${when} · 선택한 곳에서 ${Number.isFinite(t.km)?t.km.toFixed(1)+'km':''}`}));
  return box;
}

// 상세 상자 자리 후보(위·아래·오른쪽·왼쪽): 땅 위 점 q 기준, 지도 좌표
function spots(q,W,H){
  return [{l:q.x-W/2,t:q.y-78-H},{l:q.x-W/2,t:q.y+18},{l:q.x+34,t:q.y-H/2-30},{l:q.x-34-W,t:q.y-H/2-30}];
}
function boxFree(b,W,H,B){
  const xs=[b.l,b.l+W/2,b.l+W],ys=[b.t,b.t+H/2,b.t+H];
  return xs.every(x=>ys.every(y=>free(x,y,B,10)));
}
function boxSize(t){const box=t.el.querySelector('.ev-detail');return {W:box?.offsetWidth||228,H:box?.offsetHeight||(t.type==='cctv'?260:140)};}

// 선택한 곳을 화면의 같은 자리에 둔 채, 목표와 상세 상자가 다 보이는 가장 큰 줌을 찾는다
function anchoredView(t){
  const map=mapRef,B=blockers(),{W,H}=boxSize(t),start={center:map.getCenter(),zoom:map.getZoom()},f0=map.project(t.point);
  measuring=true;
  const at=z=>{map.jumpTo({center:start.center,zoom:z});const f=map.project(t.point),cs=map.project(map.getCenter());
    map.setCenter(map.unproject([cs.x+f.x-f0.x,cs.y+f.y-f0.y]));const q=map.project(t.at);
    return {center:map.getCenter(),ok:seen(q,B,40,16)&&spots(q,W,H).some(b=>boxFree(b,W,H,B))};};
  let best=null;
  if(at(start.zoom).ok)best={center:start.center,zoom:start.zoom};
  else{let lo=start.zoom-6,hi=start.zoom;for(let i=0;i<14;i++){const z=(lo+hi)/2,r=at(z);if(r.ok){lo=z;best={center:r.center,zoom:z};}else hi=z;}}
  map.jumpTo(start);measuring=false;
  return best;
}

function placeDetail(t){
  const box=t.el.querySelector('.ev-detail');if(!box)return;
  const B=blockers(),{W,H}=boxSize(t),q=mapRef.project(t.at),list=spots(q,W,H),b=list.find(x=>boxFree(x,W,H,B))||list[0];
  Object.assign(box.style,{left:(b.l-q.x)+'px',top:(b.t-q.y)+'px',bottom:'auto',transform:'none'});
}

function toggle(t){
  const map=mapRef;
  if(active===t){ // 원래 시점으로
    active=null;t.el.classList.remove('on');t.el.querySelector('.ev-detail')?.remove();links();
    if(saved)map.easeTo({...saved,duration:900,essential:true});saved=null;return;
  }
  if(active){active.el.classList.remove('on');active.el.querySelector('.ev-detail')?.remove();}
  if(!saved)saved={center:map.getCenter(),zoom:map.getZoom()};
  active=t;t.el.classList.add('on');t.el.append(detail(t));links();
  const view=anchoredView(t);
  if(view&&(Math.abs(view.zoom-map.getZoom())>.01))map.easeTo({center:view.center,zoom:view.zoom,duration:1000,essential:true});
  map.once('idle',()=>placeDetail(t));placeDetail(t);
}

// 패널의 실측값 아래에도 등급 배지
export function stationBadge(details,station){
  if(!validValue(station?.pm25))return;
  const g=gradeOf(station.pm25),value=[...details.querySelectorAll('.evidence-section')][1]?.querySelector('.value');
  if(!value)return;
  const badge=document.createElement('span');badge.className='badge';badge.textContent=g.name;badge.style.background=rgb(g.light);
  value.after(badge);
}

// 캡처용: ?evopen=station|cctv
export function openTarget(type){const t=targets.find(x=>x.type===type);if(t&&active!==t)toggle(t);}
