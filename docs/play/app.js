'use strict';

// Constants
const GLYPHS = { k: '♚', q: '♛', r: '♜', b: '♝', n: '♞', p: '♟' };
const OUTLINES = { k: '♔', q: '♕', r: '♖', b: '♗', n: '♘', p: '♙' };
const TEXT_STYLE = '︎';
const FILES = 'abcdefgh';
const STRENGTHS = {
    instinct: { name: 'Instinct', depth: 1, millis: 0, randomness: 0.03 },
    careful: { name: 'Careful', depth: 10, millis: 500, randomness: 0.02 },
    strong: { name: 'Strong', depth: 10, millis: 2500, randomness: 0.01 },
};
const ENDINGS = {
    stalemate: 'Draw by stalemate.',
    repetition: 'Draw by threefold repetition.',
    fifty_moves: 'Draw by the fifty move rule.',
    insufficient_material: 'Draw. Neither side can checkmate.',
};

const game = {
    moves: [],
    sans: [],
    legal: [],
    fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    status: 'ongoing',
    human: 'w',
    strength: 'careful',
    flipped: false,
    busy: false,
    resigned: false,
    selected: null,
    lastThink: null,
    error: null,
};

const prefs = loadPrefs();
const el = id => document.getElementById(id);

// Preferences
function loadPrefs() {
    const fallback = { color: 'random', strength: 'careful' };
    try {
        const saved = { ...fallback, ...JSON.parse(localStorage.getItem('kestrel-prefs') || '{}') };
        return STRENGTHS[saved.strength] ? saved : { ...saved, strength: fallback.strength };
    } catch {
        return fallback;
    }
}

function savePrefs() {
    try {
        localStorage.setItem('kestrel-prefs', JSON.stringify(prefs));
    } catch {
        return;
    }
}

// Engine Calls
const worker = new Worker('engine-worker.js');
const pending = new Map();
let nextCall = 1;

worker.onmessage = event => {
    const { id, ok, data, error } = event.data;
    const call = pending.get(id);
    pending.delete(id);
    if (ok) call.resolve(data);
    else call.reject(new Error(error));
};

function engine(type, args = {}) {
    return new Promise((resolve, reject) => {
        const id = nextCall++;
        pending.set(id, { resolve, reject });
        worker.postMessage({ id, type, ...args });
    });
}

// Board Helpers
function parseFen(fen) {
    const pieces = {};
    fen.split(' ')[0].split('/').forEach((row, i) => {
        let file = 0;
        for (const ch of row) {
            if (/\d/.test(ch)) { file += Number(ch); continue; }
            pieces[FILES[file] + (8 - i)] = { color: ch === ch.toUpperCase() ? 'w' : 'b', type: ch.toLowerCase() };
            file += 1;
        }
    });
    return pieces;
}

const turnOf = plies => (plies % 2 === 0 ? 'w' : 'b');
const sideToMove = () => turnOf(game.moves.length);
const isOver = () => game.resigned || game.status !== 'ongoing';
const humanToMove = () => !isOver() && !game.busy && sideToMove() === game.human;
const targetsFrom = sq => game.legal.filter(m => m.uci.slice(0, 2) === sq);

function pieceEl(piece) {
    const span = document.createElement('span');
    span.className = `piece ${piece.color}`;
    span.textContent = GLYPHS[piece.type] + TEXT_STYLE;
    if (piece.color === 'w') {
        const lines = document.createElement('span');
        lines.className = 'piece-lines';
        lines.textContent = OUTLINES[piece.type] + TEXT_STYLE;
        span.appendChild(lines);
    }
    return span;
}

// Rendering
function renderBoard() {
    const board = el('board');
    const pieces = parseFen(game.fen);
    const last = game.moves[game.moves.length - 1];
    const lastSquares = last ? [last.slice(0, 2), last.slice(2, 4)] : [];
    const lastSan = game.sans[game.sans.length - 1] || '';
    const checked = /[+#]$/.test(lastSan) ? sideToMove() : null;
    const targets = new Set(game.selected ? targetsFrom(game.selected).map(m => m.uci.slice(2, 4)) : []);

    board.replaceChildren();
    for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
            const rank = game.flipped ? row + 1 : 8 - row;
            const file = game.flipped ? 7 - col : col;
            const name = FILES[file] + rank;
            const sq = document.createElement('div');
            sq.className = `sq ${(file + rank) % 2 === 1 ? 'dark' : 'light'}`;
            sq.dataset.sq = name;
            if (lastSquares.includes(name)) sq.classList.add('last');
            if (name === game.selected) sq.classList.add('selected');
            if (targets.has(name)) sq.classList.add('target');
            const piece = pieces[name];
            if (piece) {
                sq.classList.add('occupied');
                if (checked && piece.type === 'k' && piece.color === checked) sq.classList.add('check');
                const span = pieceEl(piece);
                if (humanToMove() && piece.color === game.human) span.classList.add('movable');
                sq.appendChild(span);
            }
            if (col === 0) sq.insertAdjacentHTML('beforeend', `<span class="coord rank">${rank}</span>`);
            if (row === 7) sq.insertAdjacentHTML('beforeend', `<span class="coord file">${FILES[file]}</span>`);
            board.appendChild(sq);
        }
    }
}

function statusText() {
    if (game.error) return { text: game.error, cls: 'loss' };
    if (game.resigned) return { text: 'You resigned. Kestrel wins.', cls: 'loss' };
    if (game.status.startsWith('checkmate')) {
        const winner = game.status === 'checkmate_white' ? 'w' : 'b';
        return winner === game.human
            ? { text: 'Checkmate. You win.', cls: 'win' }
            : { text: 'Checkmate. Kestrel wins.', cls: 'loss' };
    }
    if (ENDINGS[game.status]) return { text: ENDINGS[game.status], cls: '' };
    if (game.busy) return { text: 'Kestrel is thinking', cls: 'thinking' };
    const inCheck = /[+]$/.test(game.sans[game.sans.length - 1] || '');
    return { text: inCheck ? 'Your move. You are in check.' : 'Your move.', cls: '' };
}

function renderStrips() {
    const strength = STRENGTHS[game.strength].name;
    const strip = color => {
        const human = color === game.human;
        const name = human ? 'You' : `Kestrel <span class="meta">${strength.toLowerCase()}</span>`;
        let meta = color === 'w' ? 'white' : 'black';
        if (!human && game.busy) meta = '<span class="thinking">thinking</span>';
        else if (!human && game.lastThink !== null) meta = `${meta} · ${game.lastThink.toFixed(2)}s`;
        return `<span class="name">${name}</span><span class="meta">${meta}</span>`;
    };
    const bottom = game.flipped ? 'b' : 'w';
    el('bottom-strip').innerHTML = strip(bottom);
    el('top-strip').innerHTML = strip(bottom === 'w' ? 'b' : 'w');
}

function renderMoves() {
    const list = el('moves');
    list.replaceChildren();
    if (!game.sans.length) {
        list.innerHTML = '<li class="empty">No moves yet.</li>';
        return;
    }
    game.sans.forEach((san, i) => {
        if (i % 2 === 0) list.insertAdjacentHTML('beforeend', `<li class="num">${i / 2 + 1}.</li>`);
        const li = document.createElement('li');
        li.className = 'mv' + (i === game.sans.length - 1 ? ' latest' : '');
        li.textContent = san;
        list.appendChild(li);
    });
    list.scrollTop = list.scrollHeight;
}

function render() {
    renderBoard();
    renderStrips();
    renderMoves();
    const { text, cls } = statusText();
    const status = el('status');
    status.textContent = text;
    status.className = `status ${cls === 'thinking' ? '' : cls}`;
    if (cls === 'thinking') status.innerHTML = '<span class="thinking">Kestrel is thinking</span>';
    el('undo').disabled = game.busy || !game.moves.some((_, i) => turnOf(i) === game.human);
    el('resign').disabled = game.busy || isOver() || game.moves.length < 2;
}

// Game Flow
async function refresh() {
    const data = await engine('position', { moves: game.moves });
    game.fen = data.fen;
    game.status = data.status;
    game.legal = data.legal;
    game.error = null;
}

async function guarded(fn) {
    try {
        await fn();
    } catch (err) {
        game.busy = false;
        game.error = `The engine stopped: ${err.message}.`;
    }
    render();
}

async function engineTurn() {
    game.busy = true;
    render();
    const reply = await engine('move', { moves: game.moves, strength: STRENGTHS[game.strength] });
    game.moves.push(reply.uci);
    game.sans.push(reply.san);
    game.lastThink = reply.seconds;
    game.busy = false;
    await refresh();
}

async function playMove(move) {
    game.selected = null;
    game.moves.push(move.uci);
    game.sans.push(move.san);
    render();
    await refresh();
    if (!isOver()) await engineTurn();
}

function tryMove(from, to) {
    const options = game.legal.filter(m => m.uci.slice(0, 2) === from && m.uci.slice(2, 4) === to);
    if (!options.length) return false;
    if (options.length === 1) {
        guarded(() => playMove(options[0]));
        return true;
    }
    showPromotion(options);
    return true;
}

function showPromotion(options) {
    const box = el('promotion');
    box.replaceChildren();
    for (const type of ['q', 'r', 'b', 'n']) {
        const move = options.find(m => m.uci.endsWith(type));
        if (!move) continue;
        const button = document.createElement('button');
        button.title = { q: 'Queen', r: 'Rook', b: 'Bishop', n: 'Knight' }[type];
        button.appendChild(pieceEl({ color: game.human, type }));
        button.addEventListener('click', () => {
            box.hidden = true;
            guarded(() => playMove(move));
        });
        box.appendChild(button);
    }
    box.hidden = false;
}

async function newGame() {
    const color = prefs.color === 'random' ? (Math.random() < 0.5 ? 'white' : 'black') : prefs.color;
    Object.assign(game, {
        moves: [], sans: [], legal: [], human: color === 'white' ? 'w' : 'b', strength: prefs.strength,
        resigned: false, selected: null, lastThink: null, busy: false,
    });
    game.flipped = game.human === 'b';
    el('promotion').hidden = true;
    await refresh();
    if (game.human === 'b') await engineTurn();
}

async function undo() {
    if (game.busy || !game.moves.length) return;
    game.resigned = false;
    game.selected = null;
    el('promotion').hidden = true;
    do {
        game.moves.pop();
        game.sans.pop();
    } while (game.moves.length && sideToMove() !== game.human);
    await refresh();
    if (sideToMove() !== game.human) await engineTurn();
}

// Pointer Input
const drag = { from: null, startX: 0, startY: 0, ghost: null };

function squareAt(x, y) {
    const hit = document.elementFromPoint(x, y);
    return hit && hit.closest('.sq') ? hit.closest('.sq').dataset.sq : null;
}

function onPointerDown(event) {
    const sqEl = event.target.closest('.sq');
    if (!sqEl || !humanToMove()) return;
    const sq = sqEl.dataset.sq;
    if (game.selected && targetsFrom(game.selected).some(m => m.uci.slice(2, 4) === sq)) {
        tryMove(game.selected, sq);
        return;
    }
    const piece = parseFen(game.fen)[sq];
    if (piece && piece.color === game.human) {
        game.selected = sq;
        drag.from = sq;
        drag.startX = event.clientX;
        drag.startY = event.clientY;
        render();
        event.preventDefault();
    } else {
        game.selected = null;
        render();
    }
}

function onPointerMove(event) {
    if (!drag.from) return;
    if (!drag.ghost) {
        if (Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) < 5) return;
        const source = el('board').querySelector(`[data-sq="${drag.from}"] .piece`);
        if (!source) return;
        drag.ghost = source.cloneNode(true);
        drag.ghost.classList.add('drag-ghost');
        drag.ghost.style.fontSize = getComputedStyle(source).fontSize;
        document.body.appendChild(drag.ghost);
        source.classList.add('dragging');
    }
    drag.ghost.style.left = `${event.clientX}px`;
    drag.ghost.style.top = `${event.clientY}px`;
}

function onPointerUp(event) {
    if (!drag.from) return;
    const from = drag.from;
    const wasDragging = Boolean(drag.ghost);
    if (drag.ghost) drag.ghost.remove();
    drag.from = null;
    drag.ghost = null;
    if (!wasDragging) return;
    const to = squareAt(event.clientX, event.clientY);
    if (!(to && to !== from && tryMove(from, to))) render();
}

// Controls
function setupSegmented(id, key) {
    const group = el(id);
    const sync = () => group.querySelectorAll('button').forEach(b => b.classList.toggle('active', b.dataset.value === prefs[key]));
    group.addEventListener('click', event => {
        const button = event.target.closest('button');
        if (!button) return;
        prefs[key] = button.dataset.value;
        savePrefs();
        sync();
    });
    sync();
}

function pgnText() {
    let result = '*';
    if (game.resigned) result = game.human === 'w' ? '0-1' : '1-0';
    else if (game.status === 'checkmate_white') result = '1-0';
    else if (game.status === 'checkmate_black') result = '0-1';
    else if (game.status !== 'ongoing') result = '1/2-1/2';
    const date = new Date().toISOString().slice(0, 10).replaceAll('-', '.');
    const ai = `Kestrel (${STRENGTHS[game.strength].name.toLowerCase()})`;
    const [white, black] = game.human === 'w' ? ['You', ai] : [ai, 'You'];
    const body = game.sans.map((san, i) => (i % 2 === 0 ? `${i / 2 + 1}. ${san}` : san)).join(' ');
    return `[Event "Kestrel game"]\n[Date "${date}"]\n[White "${white}"]\n[Black "${black}"]\n[Result "${result}"]\n\n${body} ${result}\n`;
}

async function copyPgn() {
    const button = el('copy-pgn');
    try {
        await navigator.clipboard.writeText(pgnText());
        button.textContent = 'copied';
    } catch {
        button.textContent = 'copy failed';
    }
    setTimeout(() => { button.textContent = 'copy PGN'; }, 1500);
}

function init() {
    setupSegmented('color-choice', 'color');
    setupSegmented('depth-choice', 'strength');
    const board = el('board');
    board.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    el('new-game').addEventListener('click', () => guarded(newGame));
    el('undo').addEventListener('click', () => guarded(undo));
    el('flip').addEventListener('click', () => { game.flipped = !game.flipped; render(); });
    el('resign').addEventListener('click', () => { game.resigned = true; game.selected = null; render(); });
    el('copy-pgn').addEventListener('click', copyPgn);
    render();
    engine('info')
        .then(info => {
            el('model-info').textContent =
                `The network here has ${info.blocks} residual blocks of ${info.channels} channels and runs in your browser.`;
        })
        .catch(() => {});
    guarded(newGame);
}

init();
