import { useEffect, useState } from "react";

export function useStoredState(key, initial, parse = (value) => value, serialize = String) {
  const [value, setValue] = useState(() => {
    try {
      const stored = localStorage.getItem(key);
      return stored == null ? initial : parse(stored);
    } catch {
      return initial;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(key, serialize(value));
    } catch {
      // Ignore localStorage failures in private or locked-down contexts.
    }
  }, [key, value, serialize]);

  return [value, setValue];
}
