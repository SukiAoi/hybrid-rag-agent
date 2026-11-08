#!/bin/bash
# 访问统计：读取 Caddy 访问日志并汇总
# 用法：/opt/rag/traffic.sh      或    ssh ubuntu@43.155.217.47 '/opt/rag/traffic.sh'
sudo docker exec caddy sh -c "cat /data/access.log" 2>/dev/null | python3 -c "
import sys, json, collections, datetime as dt
tot=0; ips=set(); bot=0
paths=collections.Counter(); ask_ip=collections.Counter()
status=collections.Counter(); first=last=None
BOT=('.env','.git','phpinfo','wp-','admin','credentials','.svn','.hg','actuator','graphql','config.json','.DS_Store','server-status','xmlrpc')
for line in sys.stdin:
    line=line.strip()
    if not line.startswith('{'): continue
    try: e=json.loads(line)
    except: continue
    r=e.get('request',{}); p=r.get('uri','?').split('?')[0]
    ip=r.get('remote_ip','?'); ts=e.get('ts',0); st=e.get('status')
    tot+=1; ips.add(ip); status[st]+=1
    if first is None or ts<first: first=ts
    if last is None or ts>last: last=ts
    if any(b in p for b in BOT): bot+=1
    else: paths[p]+=1
    if p=='/ask': ask_ip[ip]+=1
f=lambda t: dt.datetime.fromtimestamp(t).strftime('%m-%d %H:%M') if t else '-'
print(f'时间跨度   {f(first)} -> {f(last)}')
print(f'总请求     {tot}    独立 IP  {len(ips)}')
print(f'扫描尝试   {bot}  ({bot*100//max(tot,1)}%)')
print(f'正常请求   {tot-bot}')
print()
print('正常路径 TOP:')
for p,c in paths.most_common(8): print(f'  {c:>4}  {p}')
print()
print(f'提问次数   {sum(ask_ip.values())}   来自 {len(ask_ip)} 个 IP')
for ip,c in ask_ip.most_common(8): print(f'  {c:>3}  {ip}')
print()
print('状态码:', dict(status))
"
