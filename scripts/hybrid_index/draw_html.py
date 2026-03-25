def parse_ids(line: str) -> list:
    return [int(x) for x in line.split(":", 1)[1].strip().split(",")]


def save_three_column_html(
        dense_ids: list,
        sparse_ids: list,
        rerank_ids: list,
        filename: str = "result_three_col.html",
        title: str = "Hybrid Retrieval ID Check",
        meta: str = "",
):
    dense_set   = set(dense_ids)
    sparse_set  = set(sparse_ids)

    # rank 查找表（从 1 开始）
    dense_rank  = {vid: rank for rank, vid in enumerate(dense_ids,  1)}
    sparse_rank = {vid: rank for rank, vid in enumerate(sparse_ids, 1)}

    # ── 统计 ──
    k         = len(rerank_ids)
    r_both    = sum(1 for v in rerank_ids if v in dense_set and v in sparse_set)
    r_d_only  = sum(1 for v in rerank_ids if v in dense_set and v not in sparse_set)
    r_s_only  = sum(1 for v in rerank_ids if v not in dense_set and v in sparse_set)
    r_neither = sum(1 for v in rerank_ids if v not in dense_set and v not in sparse_set)

    # ── Dense 列（无高亮，纯列表）──
    dense_rows = ""
    for rank, vid in enumerate(dense_ids, 1):
        dense_rows += f"""<tr>
          <td class="rank">{rank}</td>
          <td><code>{vid}</code></td>
        </tr>\n"""

    # ── Sparse 列（无高亮，纯列表）──
    sparse_rows = ""
    for rank, vid in enumerate(sparse_ids, 1):
        sparse_rows += f"""<tr>
          <td class="rank">{rank}</td>
          <td><code>{vid}</code></td>
        </tr>\n"""

    # ── Rerank 列（三色高亮 + dense/sparse rank 标注）──
    rerank_rows = ""
    for rank, vid in enumerate(rerank_ids, 1):
        in_dense  = vid in dense_set
        in_sparse = vid in sparse_set

        if in_dense and in_sparse:
            row_cls = "both"
            tag_lbl = "dense ∩ sparse"
        elif in_dense:
            row_cls = "dense"
            tag_lbl = "dense only"
        elif in_sparse:
            row_cls = "sparse"
            tag_lbl = "sparse only"
        else:
            row_cls = "neither"
            tag_lbl = "neither"

        d_rank_str = f'<span class="rank-badge rank-d">D#{dense_rank[vid]}</span>' \
            if in_dense  else '<span class="rank-badge rank-na">D: —</span>'
        s_rank_str = f'<span class="rank-badge rank-s">S#{sparse_rank[vid]}</span>' \
            if in_sparse else '<span class="rank-badge rank-na">S: —</span>'

        rerank_rows += f"""<tr class="row-{row_cls}">
          <td class="rank">{rank}</td>
          <td><code>{vid}</code></td>
          <td><span class="tag tag-{row_cls}">{tag_lbl}</span></td>
          <td class="rank-cell">{d_rank_str}{s_rank_str}</td>
        </tr>\n"""

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', Arial, sans-serif;
      background: #f0f2f5;
      color: #333;
      padding: 28px;
    }}
    h1   {{ font-size: 20px; margin-bottom: 6px; }}
    .meta {{
      font-size: 13px; color: #666;
      margin-bottom: 18px; font-family: monospace;
    }}

    /* ── 统计卡片 ── */
    .stats {{
      display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap;
    }}
    .stat-card {{
      background: white; border-radius: 8px;
      padding: 10px 18px; box-shadow: 0 1px 4px rgba(0,0,0,.08);
      min-width: 110px; text-align: center;
    }}
    .stat-card .val {{ font-size: 24px; font-weight: 700; }}
    .stat-card .pct {{ font-size: 11px; color: #999; }}
    .stat-card .lbl {{ font-size: 11px; color: #888; margin-top: 2px; }}

    /* ── 图例 ── */
    .legend {{
      display: flex; gap: 12px; flex-wrap: wrap;
      align-items: center; font-size: 12px; margin-bottom: 16px;
    }}
    .legend-item {{ display: flex; align-items: center; gap: 5px; }}
    .legend-dot  {{ width: 12px; height: 12px; border-radius: 3px; }}

    /* ── 三列布局 ── */
    .three-col {{
      display: grid;
      grid-template-columns: 1fr 1fr 1.6fr;   /* rerank 列稍宽 */
      gap: 16px;
      align-items: start;
    }}
    .col-wrap {{
      background: white;
      border-radius: 10px;
      box-shadow: 0 1px 6px rgba(0,0,0,.08);
      overflow: hidden;
    }}
    .col-header {{
      padding: 12px 16px;
      font-size: 14px; font-weight: 700;
      color: white; text-align: center;
    }}
    .h-dense  {{ background: #2c5f9e; }}
    .h-sparse {{ background: #9e6a00; }}
    .h-rerank {{ background: #2c3e50; }}

    .scroll-body {{ max-height: 75vh; overflow-y: auto; }}

    table {{ width: 100%; border-collapse: collapse; }}
    thead th {{
      padding: 8px 10px; text-align: center;
      font-size: 12px; color: #555;
      position: sticky; top: 0; z-index: 1;
      background: #eef2f7;
      border-bottom: 1px solid #dde3ec;
    }}
    tbody tr {{ border-bottom: 1px solid #f0f0f0; transition: filter .12s; }}
    tbody tr:hover {{ filter: brightness(.93); }}
    tbody td {{ padding: 7px 10px; text-align: center; font-size: 13px; }}
    td.rank  {{ color: #bbb; font-size: 11px; width: 32px; }}
    code {{ font-family: 'Courier New', monospace; font-size: 12px; }}

    /* ── Rerank 行颜色 ── */
    tr.row-both    {{ background: #d4edda; }}
    tr.row-dense   {{ background: #cce0ff; }}
    tr.row-sparse  {{ background: #ffe8b0; }}
    tr.row-neither {{ background: #f0f0f0; }}

    /* ── 来源标签 ── */
    .tag {{
      display: inline-block; padding: 2px 8px;
      border-radius: 10px; font-size: 11px; font-weight: 600;
    }}
    .tag-both    {{ background: #82c882; color: #1a5c1a; }}
    .tag-dense   {{ background: #80b0e8; color: #0d2d6e; }}
    .tag-sparse  {{ background: #f0a830; color: #6b3d00; }}
    .tag-neither {{ background: #c0c0c0; color: #444;    }}

    /* ── Rank 徽章 ── */
    .rank-cell {{ white-space: nowrap; }}
    .rank-badge {{
      display: inline-block; margin: 1px 3px;
      padding: 2px 7px; border-radius: 8px;
      font-size: 11px; font-weight: 600;
    }}
    .rank-d  {{ background: #cce0ff; color: #0d2d6e; }}
    .rank-s  {{ background: #ffe8b0; color: #6b3d00; }}
    .rank-na {{ background: #eee;    color: #999;    }}
  </style>
</head>
<body>
  <h1>🔍 {title}</h1>
  <div class="meta">{meta}</div>

  <!-- 统计卡片（针对 rerank top-k） -->
  <div class="stats">
    <div class="stat-card" style="border-top:3px solid #555">
      <div class="val">{k}</div><div class="lbl">Rerank Top-K</div>
    </div>
    <div class="stat-card" style="border-top:3px solid #5cb85c">
      <div class="val" style="color:#3a7a3a">{r_both}</div>
      <div class="pct">{100*r_both/k:.1f}%</div>
      <div class="lbl">dense ∩ sparse</div>
    </div>
    <div class="stat-card" style="border-top:3px solid #5b9bd5">
      <div class="val" style="color:#1a4a8a">{r_d_only}</div>
      <div class="pct">{100*r_d_only/k:.1f}%</div>
      <div class="lbl">dense only</div>
    </div>
    <div class="stat-card" style="border-top:3px solid #f0a830">
      <div class="val" style="color:#7a4a00">{r_s_only}</div>
      <div class="pct">{100*r_s_only/k:.1f}%</div>
      <div class="lbl">sparse only</div>
    </div>
    <div class="stat-card" style="border-top:3px solid #aaa">
      <div class="val" style="color:#555">{r_neither}</div>
      <div class="pct">{100*r_neither/k:.1f}%</div>
      <div class="lbl">neither</div>
    </div>
  </div>

  <!-- 图例 -->
  <div class="legend">
    <span style="color:#666">Rerank 颜色：</span>
    <div class="legend-item"><div class="legend-dot" style="background:#82c882"></div>dense ∩ sparse</div>
    <div class="legend-item"><div class="legend-dot" style="background:#80b0e8"></div>dense only</div>
    <div class="legend-item"><div class="legend-dot" style="background:#f0a830"></div>sparse only</div>
    <div class="legend-item"><div class="legend-dot" style="background:#c0c0c0"></div>neither</div>
    <span style="color:#666; margin-left:10px">Rank 标注：</span>
    <div class="legend-item"><span class="rank-badge rank-d">D#n</span>在 dense 中的排名</div>
    <div class="legend-item"><span class="rank-badge rank-s">S#n</span>在 sparse 中的排名</div>
  </div>

  <!-- 三列 -->
  <div class="three-col">

    <!-- Dense（纯列表） -->
    <div class="col-wrap">
      <div class="col-header h-dense">🔵 Dense Top-{len(dense_ids)}</div>
      <div class="scroll-body">
        <table>
          <thead><tr><th>#</th><th>ID</th></tr></thead>
          <tbody>{dense_rows}</tbody>
        </table>
      </div>
    </div>

    <!-- Sparse（纯列表） -->
    <div class="col-wrap">
      <div class="col-header h-sparse">🟡 Sparse Top-{len(sparse_ids)}</div>
      <div class="scroll-body">
        <table>
          <thead><tr><th>#</th><th>ID</th></tr></thead>
          <tbody>{sparse_rows}</tbody>
        </table>
      </div>
    </div>

    <!-- Rerank（高亮 + rank 标注） -->
    <div class="col-wrap">
      <div class="col-header h-rerank">🏆 Rerank Top-{len(rerank_ids)}</div>
      <div class="scroll-body">
        <table>
          <thead>
            <tr>
              <th>#</th><th>ID</th><th>来源</th><th>原始排名</th>
            </tr>
          </thead>
          <tbody>{rerank_rows}</tbody>
        </table>
      </div>
    </div>

  </div>
</body>
</html>"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ 已保存至 {filename}")


# ── 主程序 ────────────────────────────────────────────
def main():
    txt_file = "603_hybrid_exp3_msmarco_bge_splade_ids.txt"
    dense_ids = sparse_ids = rerank_ids = []
    meta_lines = []

    with open(txt_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith("#"):
                meta_lines.append(line.lstrip("# "))
            elif line.startswith("dense_top"):
                dense_ids  = parse_ids(line)
            elif line.startswith("sparse_top"):
                sparse_ids = parse_ids(line)
            elif line.startswith("top"):
                rerank_ids = parse_ids(line)

    save_three_column_html(
        dense_ids  = dense_ids,
        sparse_ids = sparse_ids,
        rerank_ids = rerank_ids,
        filename   = "603_hybrid_exp3_msmarco_bge_splade_ids.html",
        title      = meta_lines[0] if meta_lines else "Hybrid Retrieval ID Check",
        meta       = " | ".join(meta_lines[1:]),
    )

if __name__ == "__main__":
    main()
