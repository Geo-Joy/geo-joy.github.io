/**
 * breach.guru — OG Image Generator (v3)
 *
 * Color scheme:
 *   - HEADER_COLOR (#C9A84C) → gold from blog profile ring / active nav
 *   - ACCENT (teal, per-post hash) → post title, tag pills
 *   - Tags rendered as bordered pill badges below the meta line
 *
 * Usage:
 *   node scripts/generate-og.mjs
 */

import { Resvg } from '@resvg/resvg-js';
import matter from 'gray-matter';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const POSTS_DIR = path.join(__dirname, '..', '_posts');
const OUTPUT_DIR = path.join(__dirname, '..', 'images', 'og');

if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

// ─── THEME ──────────────────────────────────────────────────────

const HEADER_COLOR = '#C9A84C'; // gold — matches blog profile ring & nav accent

const ACCENTS = [
  '#2DD4BF', // teal-400
  '#5EEAD4', // teal-300
  '#14B8A6', // teal-500
  '#06B6D4', // cyan-500
  '#22D3EE', // cyan-400
];

// ─── helpers ────────────────────────────────────────────────────

function escapeXml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function getAccent(title) {
  let hash = 0;
  for (let i = 0; i < title.length; i++) {
    hash = title.charCodeAt(i) + ((hash << 5) - hash);
  }
  return ACCENTS[Math.abs(hash) % ACCENTS.length];
}

function wrapText(title, maxLines = 3, maxChars = 26) {
  const words = title.split(/\s+/);
  const lines = [];
  let cur = '';

  for (const word of words) {
    const test = cur ? `${cur} ${word}` : word;
    if (test.length > maxChars && cur) {
      lines.push(cur);
      cur = word;
    } else {
      cur = test;
    }
  }
  if (cur) lines.push(cur);

  while (lines.length > maxLines) {
    const overflow = lines.pop();
    lines[lines.length - 1] += ' ' + overflow;
  }

  return lines;
}

function readingTime(content) {
  const words = content.replace(/---[\s\S]*?---/, '').trim().split(/\s+/).length;
  const mins = Math.max(1, Math.round(words / 230));
  return `${mins} min read`;
}

function fmtDate(d) {
  if (!(d instanceof Date)) return '';
  const months = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
  ];
  const day = String(d.getDate()).padStart(2, '0');
  return `${months[d.getMonth()]} ${day}, ${d.getFullYear()}`;
}

function loadBase64(filePath) {
  if (fs.existsSync(filePath)) {
    return fs.readFileSync(filePath, 'base64');
  }
  return null;
}

/**
 * Build tag pill badges — sized to be readable at thumbnail scale.
 * Each pill: rounded rect with subtle fill + border, centered text.
 */
function buildTagPills(tags, startX, startY, accent) {
  if (!Array.isArray(tags) || tags.length === 0) return '';

  const PILL_H = 34;
  const PILL_RX = 6;
  const PILL_PAD_X = 16;    // horizontal padding inside pill
  const PILL_GAP = 12;      // gap between pills
  const CHAR_W = 9.5;       // approximate width per char at font-size 16
  const FONT_SIZE = 16;
  const MAX_TAGS = 5;

  let x = startX;
  const pills = [];

  for (const tag of tags.slice(0, MAX_TAGS)) {
    const label = tag.toLowerCase();
    const textW = label.length * CHAR_W;
    const pillW = textW + PILL_PAD_X * 2;
    const textX = x + pillW / 2;
    const textY = startY + PILL_H / 2 + FONT_SIZE / 2 - 2;

    pills.push(`
  <rect x="${x}" y="${startY}" width="${pillW}" height="${PILL_H}" rx="${PILL_RX}"
        fill="${accent}" fill-opacity="0.07" stroke="${accent}" stroke-opacity="0.3" stroke-width="1.2"/>
  <text x="${textX}" y="${textY}" text-anchor="middle"
        font-family="'JetBrains Mono','DejaVu Sans Mono',monospace"
        font-size="${FONT_SIZE}" fill="${accent}" opacity="0.85">${escapeXml(label)}</text>`);

    x += pillW + PILL_GAP;
  }

  return pills.join('');
}

// ─── SVG builder ────────────────────────────────────────────────

function buildSVG({ title, date, tags, readTime }) {
  const accent = getAccent(title);
  const hdr = HEADER_COLOR;

  // ── title lines ──
  const fontSize = 52;
  const lineHeight = 68;
  const lines = wrapText(title, 3, 26);
  const titleX = 90;
  const titleStartY = 220;

  const tspans = lines
    .map((line, i) => {
      const y = titleStartY + i * lineHeight;
      return `<tspan x="${titleX}" y="${y}">${escapeXml(line)}</tspan>`;
    })
    .join('\n      ');

  // ── meta line (reading time only) ──
  const metaY = titleStartY + lines.length * lineHeight + 24;
  const metaStr = readTime || '';

  // ── tag pills (below meta) ──
  const pillsY = metaY + 22;
  const tagPillsSvg = buildTagPills(tags, 88, pillsY, accent);

  // ── background image (right side) ──
  const bgPath = path.join(__dirname, '..', 'assets', 'web-bg.png');
  const bgB64 = loadBase64(bgPath);

  const bgImageBlock = bgB64
    ? `
    <defs>
      <clipPath id="ornateClip">
        <rect x="620" y="84" width="540" height="500" rx="8"/>
      </clipPath>
    </defs>
    <g clip-path="url(#ornateClip)">
      <image
        href="data:image/png;base64,${bgB64}"
        x="400" y="40"
        width="1000" height="620"
        preserveAspectRatio="xMidYMid slice"
        opacity="1"
      />
    </g>
    <rect x="620" y="84" width="540" height="500" rx="8"
          fill="url(#rightFade)" opacity="0.85"/>`
    : '';

  return `<svg width="1200" height="630" xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink">

  <defs>
    <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">
      <path d="M 32 0 L 0 0 0 32" fill="none" stroke="${accent}" stroke-opacity="0.025" stroke-width="0.5"/>
    </pattern>

    <pattern id="scanlines" width="4" height="4" patternUnits="userSpaceOnUse">
      <rect width="4" height="2" fill="rgba(0,0,0,0.08)"/>
    </pattern>

    <!-- title bar — gold-tinted dark -->
    <linearGradient id="titleBarGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#1E1A10"/>
      <stop offset="100%" stop-color="#141008"/>
    </linearGradient>

    <linearGradient id="termBg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#0C1425"/>
      <stop offset="100%" stop-color="#0C1425"/>
    </linearGradient>

    <linearGradient id="rightFade" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%"  stop-color="#0C1425" stop-opacity="1"/>
      <stop offset="25%" stop-color="#0C1425" stop-opacity="0.7"/>
      <stop offset="90%" stop-color="#0C1425" stop-opacity="0.4"/>
      <stop offset="100%" stop-color="#0C1425" stop-opacity="0.6"/>
    </linearGradient>

    <!-- bottom bar: gold → teal -->
    <linearGradient id="bottomBar" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="${hdr}"/>
      <stop offset="50%" stop-color="${hdr}" stop-opacity="0.5"/>
      <stop offset="100%" stop-color="${accent}"/>
    </linearGradient>

    <filter id="glow" x="-15%" y="-15%" width="130%" height="130%">
      <feGaussianBlur stdDeviation="6" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>

    <filter id="softGlow" x="-10%" y="-10%" width="120%" height="120%">
      <feGaussianBlur stdDeviation="2.5" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>

    <filter id="goldGlow" x="-10%" y="-10%" width="120%" height="120%">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>

  <!-- ═══ BACKGROUND ═══ -->
  <rect width="1200" height="630" fill="#040810"/>
  <rect width="1200" height="630" fill="url(#grid)"/>

  <!-- ═══ TERMINAL WINDOW ═══ -->
  <rect x="40" y="30" width="1120" height="570" rx="12"
        fill="url(#termBg)"
        stroke="${accent}" stroke-opacity="0.12" stroke-width="1"/>

  <!-- ornate background figure (right side) -->
  ${bgImageBlock}

  <!-- scanlines -->
  <rect x="40" y="30" width="1120" height="570" rx="12" fill="url(#scanlines)" opacity="0.4"/>

  <!-- ═══ TITLE BAR (single path, no seam) ═══ -->
  <path d="M52,30 H1148 A12,12 0 0,1 1160,42 V72 H40 V42 A12,12 0 0,1 52,30 Z"
        fill="url(#titleBarGrad)"/>
  <rect x="40" y="72" width="1120" height="1" fill="${hdr}" opacity="0.12"/>

  <!-- traffic lights -->
  <circle cx="68" cy="51" r="6" fill="#FF5F56" stroke="#E0443E" stroke-width="0.5"/>
  <circle cx="88" cy="51" r="6" fill="#FFBD2E" stroke="#DEA123" stroke-width="0.5"/>
  <circle cx="108" cy="51" r="6" fill="#27C93F" stroke="#1AAB29" stroke-width="0.5"/>

  <!-- filename — gold -->
  <text x="600" y="55" text-anchor="middle"
        font-family="'JetBrains Mono','DejaVu Sans Mono',monospace"
        font-size="12" fill="${hdr}" opacity="0.5" letter-spacing="0.5">blog.log</text>

  <!-- ═══ BRAND (gold) ═══ -->
  <text x="90" y="120"
        font-family="'JetBrains Mono','DejaVu Sans Mono',monospace"
        font-size="16" fill="${hdr}" filter="url(#goldGlow)"
        letter-spacing="1">breach.guru</text>
  <line x1="90" y1="135" x2="250" y2="135" stroke="${hdr}" stroke-opacity="0.18" stroke-width="1"/>

  <!-- ═══ TITLE (teal) ═══ -->
  <text font-family="'JetBrains Mono','DejaVu Sans Mono',monospace"
        font-size="${fontSize}" font-weight="bold"
        fill="${accent}" filter="url(#glow)"
        letter-spacing="-0.5">
      ${tspans}
  </text>

  <!-- ═══ META ═══ -->
  ${metaStr ? `
  <text x="90" y="${metaY}"
        font-family="'JetBrains Mono','DejaVu Sans Mono',monospace"
        font-size="14" fill="#64748B" letter-spacing="0.3">${escapeXml(metaStr)}</text>` : ''}

  <!-- ═══ TAG PILLS ═══ -->
  ${tagPillsSvg}

  <!-- ═══ BOTTOM BAR ═══ -->
  <text x="90" y="555"
        font-family="'JetBrains Mono','DejaVu Sans Mono',monospace"
        font-size="20" fill="${hdr}" opacity="0.6">~$</text>
  <text x="120" y="555"
        font-family="'JetBrains Mono','DejaVu Sans Mono',monospace"
        font-size="24" fill="${hdr}">breach.guru/blog</text>


  <!-- bottom accent bar -->
  <rect x="40" y="596" width="1120" height="4" rx="2"
        fill="url(#bottomBar)" opacity="0.3"/>

</svg>`;
}

// ─── main ───────────────────────────────────────────────────────

function main() {
  const files = fs.readdirSync(POSTS_DIR).filter((f) => f.endsWith('.md'));
  let generated = 0;

  for (const file of files) {
    const filePath = path.join(POSTS_DIR, file);
    const raw = fs.readFileSync(filePath, 'utf8');
    const { data, content } = matter(raw);

    if (!data.title) {
      console.warn(`⚠  Skipping ${file}: no title in front-matter`);
      continue;
    }

    const slug = file.replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/\.md$/, '');
    const readTime = readingTime(content);

    const svg = buildSVG({
      title: data.title,
      date: data.date,
      tags: data.tags,
      readTime,
    });

    const resvg = new Resvg(svg, {
      fitTo: { mode: 'original' },
      font: {
        fontFiles: [],
        loadSystemFonts: true,
      },
    });

    const pngData = resvg.render();
    const outPath = path.join(OUTPUT_DIR, `${slug}.png`);
    fs.writeFileSync(outPath, pngData.asPng());
    generated++;
    console.log(`  ✓  images/og/${slug}.png`);
  }

  console.log(`\n  Done — ${generated} OG image${generated !== 1 ? 's' : ''} generated.\n`);
}

main();