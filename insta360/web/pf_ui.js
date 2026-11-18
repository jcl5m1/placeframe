// Shared UI for the Placeframe project pages (/placeframe/: home, dashboard, visualizer, reports, videos), all served
// by the project server (mapper/server.py). One copy, loaded by every page.
//
//   PfUI.header(element, currentId, {title?, right?})
//       the project header every page shows: Placeframe, the section tabs, the page title, a page picker
//
//   PfUI.treePicker(element, {items, selected, onSelect, storageKey, placeholder})
//       a dropdown whose items are grouped in collapsible folders (item.folder = "a/b/c"); which folders are open is
//       remembered per storageKey. items: [{id, name, folder?, meta? (HTML), active?, created? (ISO time)}]: items with
//       a creation time are listed newest first, and folders by their newest item. Returns {select, setItems, value}.
//   PfUI.mapPicker(element, opts)   the tree picker for maps
//   PfUI.pageNav(element, currentId) the tree picker listing the project's pages; choosing one opens it
(function () {
  const HOST = '28palomas-rtx3090.tail3fa327.ts.net';
  const BASE = `https://${HOST}/placeframe`;
  // The project sections, shown as tabs in every page's header.
  const SECTIONS = [
    { id: 'dashboard', name: 'Dashboard', url: `${BASE}/dashboard/` },
    { id: 'visualizer', name: 'Visualizer', url: `${BASE}/visualizer/` },
    { id: 'reports', name: 'Reports', url: `${BASE}/reports/` },
    { id: 'videos', name: 'Videos', url: `${BASE}/videos/` },
  ];
  const PAGES = [
    { id: 'home', name: 'Project home', folder: 'Project', url: `${BASE}/` },
    { id: 'dashboard', name: 'Dashboard', folder: 'Project', url: `${BASE}/dashboard/`, meta: 'uploads, maps, builds' },
    { id: 'visualizer', name: 'Map visualizer', folder: 'Project', url: `${BASE}/visualizer/`, meta: 'tailnet only' },
    { id: 'reports', name: 'Reports', folder: 'Project', url: `${BASE}/reports/`, meta: 'tailnet only; a report is shared by its link' },
    { id: 'videos', name: 'Generated videos', folder: 'Project', url: `${BASE}/videos/`, meta: 'tailnet only' },
    { id: 'api', name: 'Placeframe API docs', folder: 'Placeframe', url: `https://${HOST}/schema` },
    { id: 'grafana', name: 'Logs (Grafana)', folder: 'Placeframe', url: `https://${HOST}/grafana/`, meta: 'tailnet only' },
    { id: 'db', name: 'Database (CloudBeaver)', folder: 'Placeframe', url: `https://${HOST}/cloudbeaver/`, meta: 'tailnet only' },
  ];
  const CSS = `
  .pfui-picker { position:relative; min-width:0; }
  .pfui-picker > .pfui-btn { display:flex; align-items:center; gap:8px; width:100%; text-align:left; background:#0f1218;
    color:#e7e9ee; border:1px solid #272c36; border-radius:6px; padding:7px 10px; font:inherit; font-weight:400; cursor:pointer; }
  .pfui-btn .pfui-label { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .pfui-picker > .pfui-btn::after { content:"\\25BE"; flex:none; color:#9aa1ad; }
  .pfui-picker > .pfui-btn:hover { border-color:#4cc2ff; }
  .pfui-tree { position:absolute; z-index:50; left:0; min-width:100%; width:max-content; max-width:min(560px, 92vw); top:100%;
    margin-top:4px; max-height:60vh; overflow:auto; background:#171a21; border:1px solid #272c36; border-radius:8px;
    padding:6px 8px; box-shadow:0 8px 24px rgba(0,0,0,.5); font-size:13px; }
  .pfui-tree[hidden] { display:none; }
  .pfui-tree details { margin-left:12px; } .pfui-tree > details { margin-left:0; }
  .pfui-tree summary { cursor:pointer; color:#9aa1ad; padding:3px 0; user-select:none; }
  .pfui-item { padding:4px 8px; margin-left:12px; border-radius:5px; cursor:pointer; color:#e7e9ee; }
  .pfui-tree > .pfui-item { margin-left:0; }
  .pfui-item:hover { background:#0f1218; } .pfui-item.sel { background:rgba(76,194,255,.15); }
  .pfui-meta { color:#9aa1ad; font-size:12px; margin-left:6px; }
  .pfui-path { color:#9aa1ad; }
  .pfui-header { display:flex; align-items:center; gap:18px; flex-wrap:wrap; padding:10px 0 14px; margin-bottom:18px;
    border-bottom:1px solid #272c36; }
  .pfui-brand { font-weight:700; font-size:17px; color:#e7e9ee; text-decoration:none; letter-spacing:.2px; }
  .pfui-brand span { color:#4cc2ff; }
  .pfui-tabs { display:flex; gap:4px; flex-wrap:wrap; }
  .pfui-tabs a { color:#9aa1ad; text-decoration:none; padding:5px 10px; border-radius:6px; font-size:14px; }
  .pfui-tabs a:hover { color:#e7e9ee; background:#171a21; }
  .pfui-tabs a.on { color:#e7e9ee; background:rgba(76,194,255,.15); }
  .pfui-title { flex:1; min-width:120px; color:#e7e9ee; font-size:14px; }
  .pfui-right { display:flex; align-items:center; gap:10px; }
  .pfui-pages { min-width:200px; font-size:13px; }`;

  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  function injectCss() {
    if (document.getElementById('pfui-css')) return;
    const s = document.createElement('style'); s.id = 'pfui-css'; s.textContent = CSS; document.head.appendChild(s);
  }

  function treePicker(root, opts) {
    injectCss();
    root.classList.add('pfui-picker');
    root.innerHTML = '<button type="button" class="pfui-btn"></button><div class="pfui-tree" hidden></div>';
    const btn = root.querySelector('.pfui-btn'), tree = root.querySelector('.pfui-tree');
    const openKey = (opts.storageKey || 'pfui') + '_open';
    const stored = localStorage.getItem(openKey);
    const open = new Set(stored ? JSON.parse(stored) : (opts.defaultOpen || []));
    let items = opts.items || [], selected = opts.selected || '';

    function showSelected() {
      const it = items.find(x => x.id === selected);
      btn.innerHTML = '<span class="pfui-label">' + (it ? `${it.folder ? `<span class="pfui-path">${esc(it.folder)} / </span>` : ''}${esc(it.name)}`
        : esc(opts.placeholder || (items.length ? 'Choose…' : 'Nothing to choose'))) + '</span>';
    }
    function render() {
      // The folders on the way to the current selection are shown open (not remembered).
      const cur = items.find(x => x.id === selected), shown = new Set(open);
      (cur && cur.folder ? cur.folder.split('/').filter(Boolean) : []).reduce((p, part) => { const q = p ? p + '/' + part : part; shown.add(q); return q; }, '');
      const rootNode = { folders: {}, items: [] };
      for (const it of items) {
        let n = rootNode;
        for (const part of (it.folder || '').split('/').filter(Boolean)) n = n.folders[part] ??= { folders: {}, items: [] };
        n.items.push(it);
      }
      const count = n => n.items.length + Object.values(n.folders).reduce((a, f) => a + count(f), 0);
      const newest = n => [...n.items.map(it => it.created || ''), ...Object.values(n.folders).map(newest)].reduce((a, b) => (b > a ? b : a), '');
      // Newest first (items with a creation time; others keep alphabetical order).
      const byNewest = (a, b) => (b.created || '').localeCompare(a.created || '') || a.name.localeCompare(b.name);
      const draw = (n, path) => Object.keys(n.folders).sort((a, b) => newest(n.folders[b]).localeCompare(newest(n.folders[a])) || a.localeCompare(b)).map(name => {
        const p = path ? path + '/' + name : name, f = n.folders[name];
        return `<details data-path="${esc(p)}"${shown.has(p) ? ' open' : ''}><summary>📁 ${esc(name)} <span class="pfui-meta">${count(f)}</span></summary>${draw(f, p)}</details>`;
      }).join('') + n.items.slice().sort(byNewest).map(it =>
        `<div class="pfui-item${it.id === selected ? ' sel' : ''}" data-id="${esc(it.id)}">${esc(it.name)}${it.meta ? `<span class="pfui-meta">${it.meta}</span>` : ''}</div>`).join('');
      tree.innerHTML = draw(rootNode, '') || '<span class="pfui-meta">Nothing here</span>';
      tree.querySelectorAll('.pfui-item').forEach(el => el.onclick = () => select(el.dataset.id));
      tree.querySelectorAll('details').forEach(d => d.ontoggle = () => {
        d.open ? open.add(d.dataset.path) : open.delete(d.dataset.path);
        localStorage.setItem(openKey, JSON.stringify([...open]));
      });
    }
    function select(id, fire = true) {
      selected = id; showSelected(); tree.hidden = true;
      if (fire && opts.onSelect) opts.onSelect(id);
    }
    btn.onclick = () => { tree.hidden = !tree.hidden; if (!tree.hidden) render(); };
    document.addEventListener('click', e => { if (!root.contains(e.target)) tree.hidden = true; });
    showSelected();
    return {
      select,
      setItems(next) { items = next; showSelected(); if (!tree.hidden) render(); },
      get value() { return selected; },
    };
  }

  const mapPicker = (root, opts) => treePicker(root, { storageKey: 'pfui_maps', placeholder: 'Choose a map…', ...opts });

  function pageNav(root, currentId) {
    return treePicker(root, {
      items: PAGES, selected: currentId, storageKey: 'pfui_pages', defaultOpen: [...new Set(PAGES.map(p => p.folder))],
      onSelect: id => { const p = PAGES.find(x => x.id === id); if (p && id !== currentId) location.href = p.url; },
    });
  }

  function header(root, currentId, opts = {}) {
    injectCss();
    root.className = 'pfui-header';
    root.innerHTML = `<a class="pfui-brand" href="${BASE}/">Place<span>frame</span></a>
      <nav class="pfui-tabs">${SECTIONS.map(s => `<a href="${s.url}" class="${s.id === currentId ? 'on' : ''}">${esc(s.name)}</a>`).join('')}</nav>
      <div class="pfui-title">${opts.title ? esc(opts.title) : ''}</div>
      <div class="pfui-right">${opts.right || ''}<span class="pfui-pages"></span></div>`;
    pageNav(root.querySelector('.pfui-pages'), currentId);
    return root;
  }

  window.PfUI = { treePicker, mapPicker, pageNav, header, esc, PAGES, SECTIONS, BASE };
})();
