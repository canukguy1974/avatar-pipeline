// web/components/FakeAssistant.tsx
"use client";
import { useState } from "react";

function* fakeStream(text: string) {
  for (const w of text.split(" ")) yield w + " ";
}

async function streamInto(setter: (s: string)=>void, text: string, wpm=230) {
  let out = "";
  const delay = Math.max(10, Math.floor(60000/(wpm*4)));
  for (const tok of fakeStream(text)) {
    out += tok; setter(out);
    await new Promise(r => setTimeout(r, delay));
  }
}

export default function FakeAssistant() {
  const [input, setInput] = useState("");
  const [resp, setResp] = useState("");

  const speak = (text: string) => {
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.02; u.pitch = 1.0; u.volume = 1.0;
    window.speechSynthesis.cancel(); window.speechSynthesis.speak(u);
  };

  const canned = [
    "I’m InfiniteTalk — a live avatar that listens and speaks back in real time.",
    "I watch, think, and reply with natural voice so your demo feels live.",
    "Ask me anything — I’ll answer instantly while I stay animated on screen."
  ];

  const onSend = async () => {
    const text = canned[Math.floor(Math.random()*canned.length)];
    setResp(""); await streamInto(setResp, text, 240); speak(text);
  };

  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        <input
          className="flex-1 px-3 py-2 rounded bg-zinc-900 border border-zinc-700"
          value={input}
          onChange={(e)=>setInput(e.target.value)}
          placeholder="Tell me about yourself in 1 sentence."
        />
        <button onClick={onSend}
          className="px-4 py-2 rounded bg-emerald-600 hover:bg-emerald-500">Send</button>
      </div>
      <div className="p-3 rounded bg-black/50 border border-zinc-800 min-h-[88px] whitespace-pre-wrap">
        {resp || "…"}
      </div>
    </div>
  );
}
