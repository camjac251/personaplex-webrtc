import { useEffect, useRef, useState } from "react";

export function useStoredState(key, initial, parse = (value) => value, serialize = String) {
  const [value, setValue] = useState(() => {
    try {
      const stored = localStorage.getItem(key);
      return stored == null ? initial : parse(stored);
    } catch {
      return initial;
    }
  });

  // First-render default. A value equal to it is DELETED from storage
  // rather than written, so a browser sitting at the defaults keeps
  // tracking future default changes with no version bookkeeping; only
  // deviations persist.
  const initialRef = useRef(initial);

  // Last string handed to setItem. Several call sites pass inline arrow
  // serializers recreated every render, which re-fires the effect at the
  // app's render rate; comparing the serialized form skips the redundant
  // synchronous writes.
  const lastWrittenRef = useRef(null);

  useEffect(() => {
    try {
      const next = serialize(value);
      if (next === lastWrittenRef.current) return;
      lastWrittenRef.current = next;
      if (next === serialize(initialRef.current)) {
        localStorage.removeItem(key);
        return;
      }
      localStorage.setItem(key, next);
    } catch {
      // Ignore localStorage failures in private or locked-down contexts.
    }
  }, [key, value, serialize]);

  return [value, setValue];
}
