'use strict';

// Kestrel In A Worker
const encoder = new TextEncoder();
const decoder = new TextDecoder();
let exports = null;
let info = null;

const ready = (async () => {
    const [wasm, weights] = await Promise.all([
        fetch('kestrel_web.wasm').then(r => { if (!r.ok) throw new Error(`engine download failed (${r.status})`); return r.arrayBuffer(); }),
        fetch('model.kstn').then(r => { if (!r.ok) throw new Error(`model download failed (${r.status})`); return r.arrayBuffer(); }),
    ]);
    const { instance } = await WebAssembly.instantiate(wasm, { env: { now_ms: () => performance.now() } });
    exports = instance.exports;
    const bytes = new Uint8Array(weights);
    const ptr = exports.alloc(bytes.length);
    new Uint8Array(exports.memory.buffer, ptr, bytes.length).set(bytes);
    const failed = exports.load(ptr, bytes.length, 1);
    exports.dealloc(ptr, bytes.length);
    if (failed) throw new Error(read(failed));
    const header = new DataView(weights);
    info = { blocks: header.getUint32(8, true), channels: header.getUint32(12, true), bytes: bytes.length };
})();

// Memory Helpers
function put(text) {
    const bytes = encoder.encode(text);
    const ptr = exports.alloc(Math.max(bytes.length, 1));
    new Uint8Array(exports.memory.buffer, ptr, bytes.length).set(bytes);
    return [ptr, bytes.length];
}

function release([ptr, len]) {
    exports.dealloc(ptr, Math.max(len, 1));
}

function read(len) {
    return decoder.decode(new Uint8Array(exports.memory.buffer, exports.output(), len));
}

function call(name, moves, ...rest) {
    const fen = put('startpos');
    const list = put(moves.join(' '));
    try {
        return read(exports[name](fen[0], fen[1], list[0], list[1], ...rest));
    } finally {
        release(fen);
        release(list);
    }
}

// Engine Requests
function position(moves) {
    const text = call('state', moves);
    if (text.startsWith('error')) throw new Error(text.slice(6));
    const fields = Object.fromEntries(text.split('\n').map(line => {
        const space = line.indexOf(' ');
        return [line.slice(0, space), line.slice(space + 1)];
    }));
    const legal = fields.legal ? fields.legal.split(',').map(entry => {
        const [uci, san] = entry.split(' ');
        return { uci, san };
    }) : [];
    return { fen: fields.fen, status: fields.status, legal };
}

function move(moves, strength) {
    const before = position(moves);
    const start = performance.now();
    const seed = Math.floor(Math.random() * 2 ** 31);
    const text = call('think', moves, strength.depth, strength.millis, strength.randomness, seed);
    if (text.startsWith('error') || text === 'none') throw new Error(text);
    const [uci, score, depth] = text.split(' ');
    const san = before.legal.find(m => m.uci === uci)?.san || uci;
    return { uci, san, seconds: (performance.now() - start) / 1000, score: Number(score), depth: Number(depth) };
}

self.onmessage = async event => {
    const { id, type, moves, strength } = event.data;
    try {
        await ready;
        const data = type === 'info' ? info : type === 'position' ? position(moves) : move(moves, strength);
        self.postMessage({ id, ok: true, data });
    } catch (error) {
        self.postMessage({ id, ok: false, error: error.message || String(error) });
    }
};
