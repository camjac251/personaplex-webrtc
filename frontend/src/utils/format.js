export const cls = (...parts) => parts.filter(Boolean).join(" ");

export const fmt = (value, digits = 2) => Number(value).toFixed(digits);
