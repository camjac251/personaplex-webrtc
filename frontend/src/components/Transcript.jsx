import { useLayoutEffect, useRef, useState } from "react";
import { PARAM_INFO } from "../data/dashboardData.jsx";
import { cls } from "../utils/format.js";
import { changedTranscriptLineIds, isNearTranscriptBottom, parseVocalEvents, transcriptOffset } from "../utils/transcript.js";

export function TranscriptTurn({ turn, sessionStart, live }) {
  const system = turn.role === "system" || turn.role === "vision";
  const active = live && turn.active;
  return (
    <div className={cls("line", turn.role, system && "system", active && "live", turn.audioOnly && "audio-only")}>
      <div className="turn-heading">
        <span className="who">{turn.role === "ai" ? "AI" : turn.role === "you" ? "You" : turn.role === "vision" ? "Vision" : "System"}</span>
        <time className="turn-time">{transcriptOffset(turn.at, sessionStart)}</time>
        {active && <span className="turn-live">speaking</span>}
      </div>
      <div className={cls("text", turn.audioOnly && "muted")}>
        {turn.audioOnly ? "spoke · audio only" : system ? turn.text : parseVocalEvents(turn.text).map((piece) => piece.kind === "event" ? (
          <span key={piece.at} className="vocal-event" role="img" aria-label={`Vocal event: ${piece.text}`}>{piece.text}</span>
        ) : <span key={piece.at}>{piece.text}</span>)}
        {active && <span className="typing" aria-hidden="true" />}
      </div>
    </div>
  );
}

export function Transcript({ turns, sessionStart, live }) {
  const streamRef = useRef(null);
  const previousRef = useRef([]);
  const followingRef = useRef(true);
  const jumpingRef = useRef(false);
  const unseenRef = useRef(new Set());
  const [following, setFollowing] = useState(true);
  const [unseen, setUnseen] = useState(0);

  useLayoutEffect(() => {
    const stream = streamRef.current;
    if (!stream) return;
    if (!turns.length) {
      followingRef.current = true;
      jumpingRef.current = false;
      setFollowing(true);
    }
    if (followingRef.current) {
      stream.scrollTop = stream.scrollHeight;
      unseenRef.current.clear();
      setUnseen(0);
    } else {
      for (const id of changedTranscriptLineIds(previousRef.current, turns)) unseenRef.current.add(id);
      setUnseen(unseenRef.current.size);
    }
    previousRef.current = turns;
  }, [turns]);

  useLayoutEffect(() => {
    const stream = streamRef.current;
    if (!stream || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => {
      if (followingRef.current && !jumpingRef.current) stream.scrollTop = stream.scrollHeight;
    });
    observer.observe(stream);
    return () => observer.disconnect();
  }, []);

  const handleScroll = () => {
    const near = isNearTranscriptBottom(streamRef.current);
    if (jumpingRef.current && !near) return;
    jumpingRef.current = false;
    followingRef.current = near;
    setFollowing(near);
    if (near) {
      unseenRef.current.clear();
      setUnseen(0);
    }
  };
  const cancelJump = () => { jumpingRef.current = false; };
  const jumpToLive = () => {
    followingRef.current = true;
    jumpingRef.current = true;
    setFollowing(true);
    unseenRef.current.clear();
    setUnseen(0);
    streamRef.current.scrollTo({ top: streamRef.current.scrollHeight, behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth" });
  };

  return (
    <div className="transcript-body">
      {/* biome-ignore lint/a11y/noNoninteractiveTabindex: the scrollable transcript must be keyboard-focusable to browse earlier turns */}
      <section className="transcript-stream" ref={streamRef} aria-label="Conversation transcript" tabIndex={0} onScroll={handleScroll} onWheel={cancelJump} onTouchStart={cancelJump} onKeyDown={cancelJump}>
        {turns.length === 0 ? (
          <div className="transcript-empty">
            <div>
              <div className="label">{live ? "Listening" : "No active transcript"}</div>
              <div className="sub">{live ? "Speak into your microphone." : "Configure persona on the left, then connect."}</div>
            </div>
          </div>
        ) : turns.map((turn) => <TranscriptTurn key={turn.id} turn={turn} sessionStart={sessionStart} live={live} />)}
      </section>
      {!following && (
        <button type="button" className="jump-to-live" onClick={jumpToLive} aria-label={`Jump to live. ${unseen} new or updated ${unseen === 1 ? "line" : "lines"}. Resume following the transcript.`} title={PARAM_INFO.jumpToLive.body}>
          Jump to live{unseen > 0 && <span className="jump-count">{unseen} new</span>}
        </button>
      )}
    </div>
  );
}
