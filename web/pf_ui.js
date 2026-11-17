// Shared UI for the Placeframe project pages (mapper dashboard, map visualizer). One copy, loaded by every page:
// the mapper serves it at /mapper/pf_ui.js and the visualizer's folder links to this file.
//
//   PfUI.treePicker(element, {items, selected, onSelect, storageKey, placeholder})
//       a dropdown whose items are grouped in collapsible folders (item.folder = "a/b/c"); which folders are open is
//       remembered per storageKey. items: [{id, name, folder?, meta? (HTML), active?}]. Returns {select, setItems, value}.
//   PfUI.mapPicker(element, opts)   the tree picker for maps
//   PfUI.pageNav(element, currentId) the tree picker listing the project's pages; choosing one opens it
(function () {
  const HOST = '28palomas-rtx3090.tail3fa327.ts.net';
  const PAGES = [
    { id: 'dashboard', name: 'Mapper dashboard', folder: 'Project', url: `https://${HOST}/mapper/`, meta: 'uploads, jobs, compute' },
    { id: 'viewer', name: 'Map visualizer', folder: 'Project', url: `https://${HOST}:8130/`, meta: 'tailnet only' },
    { id: 'cost', name: 'Map cost study report', folder: 'Project', url: `https://${HOST}/mapper/cost/` },
    { id: 'videos', name: 'Generated videos', folder: 'Project', url: `https://${HOST}/mapper/videos/`, meta: 'tailnet only' },
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
  .pfui-path { color:#9aa1ad; }`;

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
      const draw = (n, path) => Object.keys(n.folders).sort().map(name => {
        const p = path ? path + '/' + name : name, f = n.folders[name];
        return `<details data-path="${esc(p)}"${shown.has(p) ? ' open' : ''}><summary>📁 ${esc(name)} <span class="pfui-meta">${count(f)}</span></summary>${draw(f, p)}</details>`;
      }).join('') + n.items.slice().sort((a, b) => ((b.active ? 1 : 0) - (a.active ? 1 : 0)) || a.name.localeCompare(b.name)).map(it =>
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

  window.PfUI = { treePicker, mapPicker, pageNav, esc, PAGES };
})();
