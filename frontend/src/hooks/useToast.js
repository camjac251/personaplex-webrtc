import { useCallback, useRef } from "react";

export function useToast() {
  const timer = useRef(null);
  return useCallback((message) => {
    const element = document.getElementById("toast");
    if (!element) return;
    clearTimeout(timer.current);
    element.textContent = message;
    element.classList.add("show");
    timer.current = setTimeout(() => element.classList.remove("show"), 2500);
  }, []);
}
