"""Fixed browser observer INSIDE the OCI sandbox; never runs on the host.

No model-provided selectors, eval code, URL or browser flags. Source bytes are
served only from the verified private snapshot. Report measurements, not taste.
"""
import base64
import hashlib
import json
import mimetypes
import sys
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from playwright.sync_api import sync_playwright


METRICS = r"""() => {
 const visible=e=>{for(let p=e;p;p=p.parentElement){const s=getComputedStyle(p);if(s.display==='none'||s.visibility==='hidden'||+s.opacity===0)return false;}return e.getBoundingClientRect().width>0;};
 const selector=e=>{const parts=[];for(let p=e;p&&p!==document.body&&parts.length<4;p=p.parentElement){
   if(p.id){parts.unshift('#'+CSS.escape(p.id));break;}
   let s=p.localName;const cls=Array.from(p.classList).slice(0,2);if(cls.length)s+='.'+cls.map(CSS.escape).join('.');
   else if(p.parentElement){const same=Array.from(p.parentElement.children).filter(c=>c.localName===p.localName);if(same.length>1)s+=':nth-of-type('+(same.indexOf(p)+1)+')';}
   parts.unshift(s);
 }return parts.join(' > ').slice(0,240);};
 // Hidden translations, closed controls and assistive-only labels are not
 // broken content. Do not blanket-exempt arbitrary opacity:0 animations.
 const intentional=e=>{
   for(let p=e;p;p=p.parentElement){
     if(p.hidden||p.inert||p.getAttribute('aria-hidden')==='true')return true;
     if(p.localName==='dialog'&&!p.open)return true;
     if(p.localName==='details'&&!p.open){const summary=p.querySelector(':scope > summary');if(!summary||!summary.contains(e))return true;}
     if(p.id&&Array.from(document.querySelectorAll('[aria-controls]')).some(c=>
       (c.getAttribute('aria-expanded')==='false'||(c.getAttribute('role')==='tab'&&c.getAttribute('aria-selected')==='false'))&&
       c.getAttribute('aria-controls').split(/\s+/).includes(p.id)))return true;
     const s=getComputedStyle(p),r=p.getBoundingClientRect();
     if(['absolute','fixed'].includes(s.position)&&r.width<=1&&r.height<=1&&
        (s.clip!=='auto'||s.clipPath!=='none')&&['hidden','clip'].includes(s.overflow))return true;
   }return false;
 };
 const hidden=Array.from(document.querySelectorAll('h1,h2,h3,p,article,section')).filter(e=>e.textContent.trim()&&!visible(e));
 const color=value=>{const m=value.match(/^rgba?\(([^)]+)\)$/);if(!m)return null;const v=m[1].split(/[,\s/]+/).map(Number);return v.length>=3?[...v.slice(0,3),v[3]??1]:null;};
 const blend=(front,back)=>front.slice(0,3).map((v,i)=>v*front[3]+back[i]*(1-front[3]));
 const luminance=rgb=>rgb.map(v=>{v/=255;return v<=.04045?v/12.92:((v+.055)/1.055)**2.4;}).reduce((sum,v,i)=>sum+v*[.2126,.7152,.0722][i],0);
 const contrast=e=>{
   // Only make a hard finding for an unambiguous solid background. Artwork,
   // gradients, overlays and shadows need visual review, not guessed pixels.
   const chain=[];for(let p=e;p;p=p.parentElement)chain.push(p);
   let bg=[255,255,255];
   for(const p of chain.reverse()){
     const s=getComputedStyle(p),c=color(s.backgroundColor);
     if(!c||s.backgroundImage!=='none'||s.filter!=='none'||s.mixBlendMode!=='normal'||+s.opacity!==1)return null;
     bg=blend(c,bg);
   }
   // A positioned decorative layer may be the real backdrop.
   if(chain.some(p=>Array.from(p.children).some(child=>{
     if(child===e||child.contains(e))return false;
     const s=getComputedStyle(child),r=child.getBoundingClientRect();return ['absolute','fixed'].includes(s.position)&&r.width>0&&r.height>0&&visible(child);
   })))return null;
   const s=getComputedStyle(e),fg=color(s.color);if(!fg)return null;
   const a=luminance(blend(fg,bg)),b=luminance(bg),ratio=(Math.max(a,b)+.05)/(Math.min(a,b)+.05);
   const large=parseFloat(s.fontSize)>=24||(parseFloat(s.fontSize)>=18.66&&parseInt(s.fontWeight)>=700);
   return {selector:selector(e),ratio:Math.round(ratio*100)/100,minimum:large?3:4.5};
 };
 const unreadable=Array.from(document.querySelectorAll('h1,h2,h3')).filter(e=>visible(e)&&!intentional(e))
   .flatMap(e=>{const item=contrast(e);return item&&item.ratio<item.minimum?[item]:[];}).slice(0,10);
 const pageOverflow=document.documentElement.scrollWidth>innerWidth+1;
 const overflow=pageOverflow?Array.from(document.querySelectorAll('body *')).filter(e=>{
   // SVG path bounds are local artwork, often intentionally outside a clipped
   // viewBox. Diagnose its outer layout container, not every rect/path/circle.
   if(!visible(e))return false;
   if(e instanceof SVGElement&&e.ownerSVGElement){
     for(let svg=e.ownerSVGElement;svg;svg=svg.ownerSVGElement){
       if(['hidden','clip','auto','scroll'].includes(getComputedStyle(svg).overflowX))return false;
     }
   }
   const r=e.getBoundingClientRect();let right=r.right;
   for(let p=e.parentElement;p&&p!==document.documentElement;p=p.parentElement){
     const s=getComputedStyle(p);if(['hidden','clip','auto','scroll'].includes(s.overflowX)||s.contain.includes('paint'))right=Math.min(right,p.getBoundingClientRect().right);
   }
   return right>innerWidth+2;
 }).slice(0,8):[];
 return {title:document.title.slice(0,160), horizontal_overflow:pageOverflow,
   page_width:document.documentElement.scrollWidth,
   viewport_width:innerWidth, page_height:document.documentElement.scrollHeight,
   headings:Array.from(document.querySelectorAll('h1,h2,h3')).slice(0,25).map(e=>({text:e.textContent.slice(0,120),visible:visible(e)})),
   hidden_text_blocks:hidden.slice(0,20).map(selector),
   unexpected_hidden_text_blocks:hidden.filter(e=>!intentional(e)).slice(0,20).map(selector),
   unreadable_headings:unreadable,
   broken_images:Array.from(document.images).filter(e=>!e.complete||e.naturalWidth===0).slice(0,15).map(e=>({alt:e.alt.slice(0,80),src:(e.getAttribute('src')||'').slice(0,120)})),
   missing_anchors:Array.from(document.querySelectorAll('a[href^="#"]')).filter(e=>e.hash.length>1&&!document.getElementById(decodeURIComponent(e.hash.slice(1)))).slice(0,15).map(e=>e.hash.slice(0,120)),
   overflowing_elements:overflow.map(selector),
   overflow_details:overflow.slice(0,4).map(e=>{const r=e.getBoundingClientRect(),s=getComputedStyle(e);return {
     selector:selector(e),tag:e.localName,id:e.id.slice(0,100),classes:Array.from(e.classList).slice(0,3).map(c=>c.slice(0,100)),
     left:Math.round(r.left),right:Math.round(r.right),width:Math.round(r.width),
     css_width:s.width,min_width:s.minWidth,display:s.display,parent:e.parentElement?selector(e.parentElement):null};}),
   visible_text_chars:document.body.innerText.length};
}"""


def main():
    request = json.loads(Path("/input/request.json").read_bytes())
    sources = {}
    for item in json.loads(Path("/input/files.json").read_bytes()):
        data = (Path("/input") / item["file"]).read_bytes()
        if len(data) != item["size_bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("source integrity mismatch")
        sources[item["path"]] = data
    source = sources[request["entrypoint"]]
    reports, files = [], []
    with sync_playwright() as p:
        # Chromium's nested sandbox is unavailable under this locked-down OCI
        # profile. The outer network-none/unprivileged/read-only container is
        # mandatory and is the execution boundary, same as Python/Node jobs.
        browser = p.chromium.launch(executable_path="/usr/bin/chromium", chromium_sandbox=False,
                                   args=["--disable-dev-shm-usage", "--disable-gpu", "--renderer-process-limit=2", "--num-raster-threads=2"], timeout=20000)
        scenarios = [(v, True) for v in request["viewports"]]
        if request["check_without_scripts"]:
            scenarios.append((request["viewports"][0], False))
        for index, (viewport, scripts) in enumerate(scenarios):
            context = browser.new_context(viewport=viewport, java_script_enabled=scripts, service_workers="block", accept_downloads=False)
            errors, denied = [], []
            def route(r):
                url = urlsplit(r.request.url)
                name = unquote(url.path.lstrip("/"))
                if url.scheme == "https" and url.netloc == "workspace.invalid" and name in sources:
                    r.fulfill(body=sources[name], content_type=mimetypes.guess_type(name)[0] or "application/octet-stream")
                else:
                    if len(denied) < 10:
                        denied.append(r.request.url[:160])
                    r.abort()
            context.route("**/*", route)
            context.route_web_socket("**/*", lambda ws: ws.close())
            page = context.new_page()
            page.set_default_timeout(10000)
            page.on("pageerror", lambda e: errors.append(str(e)[:250]) if len(errors) < 10 else None)
            page.on("dialog", lambda dialog: dialog.dismiss())
            page.goto("https://workspace.invalid/" + quote(request["entrypoint"], safe="/"), wait_until="load", timeout=15000)
            # Exercise intersection-based progressive rendering without skipping
            # sections; this is bounded even for pathological page heights.
            for y in range(0, min(page.evaluate("document.documentElement.scrollHeight"), 16000), max(200, viewport["height"] // 2)):
                page.evaluate("y => window.scrollTo(0,y)", y)
                page.wait_for_timeout(40)
            page.evaluate("window.scrollTo(0,0)")
            page.wait_for_timeout(250)
            metrics = page.evaluate(METRICS)
            metrics.update(viewport=viewport, javascript=scripts, console_errors=errors, blocked_requests=denied)
            reports.append(metrics)
            png = page.screenshot(full_page=False, timeout=10000)
            files.append({"path": request["outputs"][index], "data": base64.b64encode(png).decode(),
                          "sha256": hashlib.sha256(png).hexdigest(), "size_bytes": len(png)})
            context.close()
        browser.close()
    return {"exit_code": 0, "log": "Browser inspection completed; source files were not modified.", "files": files,
            "inspection": {"source_sha256": hashlib.sha256(source).hexdigest(), "scenarios": reports,
                           "limits": "offline; no backend/API validation; viewport screenshots; measurements are not a visual design approval"}}


if __name__ == "__main__":
    try:
        result = main()
    except Exception as error:
        result = {"exit_code": 1, "log": str(error)[:4000], "files": []}
    sys.stdout.write(json.dumps(result, separators=(",", ":")))
